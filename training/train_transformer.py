import argparse
import os
import sys
from collections import deque

# allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import torch
from tqdm import trange

from agents.transformer_rnd import TransformerRNDAgent, preprocess_observation
from analysis.gradcam import GradCAM
from analysis.lrp import generate_lrp_map
from procgen import ProcgenEnv
from utils.logger import Logger


def train_transformer_rnd(
    env_name="coinrun",
    num_steps=1000000,
    device="cuda",
    num_envs=8,
    rollout_length=None,
    **agent_kwargs,
):
    env = ProcgenEnv(num_envs=num_envs, env_name=env_name, render_mode="rgb_array")
    agent = TransformerRNDAgent(env, device=torch.device(device), **agent_kwargs)

    logger = Logger(agent_name="transformer_rnd")
    gradcam = GradCAM(agent.policy, target_layer_name="encoder.conv.4")

    logger.log_config(
        {
            "env": env_name,
            "device": device,
            "num_steps": num_steps,
            "num_envs": num_envs,
            "gamma": agent.gamma,
            "lr": agent.optimizer.param_groups[0]["lr"],
            "eps_clip": agent.eps_clip,
            "update_epochs": agent.update_epochs,
            "batch_size": agent.batch_size,
            "rollout_length": rollout_length or agent.batch_size,
            "int_coef": agent.int_coef,
            "transformer_layers": agent_kwargs.get("transformer_layers", 4),
            "embed_dim": agent_kwargs.get("embed_dim", 128),
            "num_heads": agent_kwargs.get("num_heads", 8),
        }
    )

    obs_buf, actions_buf, logp_buf, value_buf, reward_buf, intrinsic_buf, done_buf = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    obs = env.reset()
    episode_rewards = np.zeros(num_envs)
    episode_intrinsic = np.zeros(num_envs)
    reward_queue = deque(maxlen=100)

    rollout_length = rollout_length or agent.batch_size
    pbar = trange(num_steps, desc="Training Transformer-RND", dynamic_ncols=True)
    for step in pbar:
        actions, logps, entropies, values, intrinsic_rewards = agent.select_action(
            obs["rgb"]
        )
        next_obs, rewards, dones, infos = env.step(np.array(actions))

        if step % 1000 == 0:
            logger.flush()

        for i in range(num_envs):
            total_r = rewards[i] + agent.int_coef * intrinsic_rewards[i]
            obs_buf.append(obs["rgb"][i])
            actions_buf.append(actions[i])
            logp_buf.append(logps[i])
            value_buf.append(values[i])
            reward_buf.append(total_r)
            intrinsic_buf.append(intrinsic_rewards[i])
            done_buf.append(dones[i])

            episode_rewards[i] += rewards[i]
            episode_intrinsic[i] += intrinsic_rewards[i]

            logger.log_scalar("extrinsic_reward", rewards[i], step)
            logger.log_scalar("intrinsic_reward", intrinsic_rewards[i], step)
            logger.log_scalar("total_reward", total_r, step)

        obs = next_obs

        if len(obs_buf) >= rollout_length:
            returns, advs = [], []
            R = 0
            for r, d in zip(reversed(reward_buf), reversed(done_buf)):
                R = r + agent.gamma * R * (1 - d)
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32)
            values_t = torch.tensor(value_buf, dtype=torch.float32)
            advs = returns - values_t

            for i in range(len(obs_buf)):
                agent.store(
                    (
                        obs_buf[i],
                        actions_buf[i],
                        logp_buf[i],
                        returns[i].item(),
                        advs[i].item(),
                        intrinsic_buf[i],
                    )
                )

            actor_loss, critic_loss, rnd_loss = agent.update()
            logger.log_scalar("actor_loss", actor_loss, step)
            logger.log_scalar("critic_loss", critic_loss, step)
            logger.log_scalar("rnd_loss", rnd_loss, step)
            (
                obs_buf,
                actions_buf,
                logp_buf,
                value_buf,
                reward_buf,
                intrinsic_buf,
                done_buf,
            ) = ([], [], [], [], [], [], [])

        for i in range(num_envs):
            if dones[i]:
                reward_queue.append(episode_rewards[i])
                logger.log_scalar("episode_reward", episode_rewards[i], step)
                logger.log_scalar(
                    "episode_intrinsic_reward", episode_intrinsic[i], step
                )
                avg = sum(reward_queue) / len(reward_queue)
                pbar.set_postfix(avg_reward=f"{avg:.2f}")
                episode_rewards[i] = 0
                episode_intrinsic[i] = 0

        if step % 5000 == 0:
            frame = preprocess_observation(obs["rgb"][0]).unsqueeze(0).to(agent.device)
            cam = gradcam.generate_cam(frame)

            img = (frame.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(
                np.uint8
            )
            heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
            sf = 4
            img_res = cv2.resize(
                img,
                (img.shape[1] * sf, img.shape[0] * sf),
                interpolation=cv2.INTER_NEAREST,
            )
            heat_res = cv2.resize(
                heat, img_res.shape[:2][::-1], interpolation=cv2.INTER_LINEAR
            )
            overlay = cv2.addWeighted(heat_res, 0.4, img_res, 0.6, 0)

            cam_dir = os.path.join("cam_outputs", "transformer_rnd")
            os.makedirs(cam_dir, exist_ok=True)
            cv2.imwrite(os.path.join(cam_dir, f"cam_step{step}.png"), overlay)

            frame_dir = os.path.join("frames", "transformer_rnd")
            os.makedirs(frame_dir, exist_ok=True)
            cv2.imwrite(os.path.join(frame_dir, f"frame_step{step}.png"), img_res)

            # ---- LRP ----
            relevance_map = generate_lrp_map(agent.policy, frame)
            relevance_map = relevance_map.sum(axis=0)
            relevance_map = (relevance_map - relevance_map.min()) / (
                relevance_map.max() - relevance_map.min() + 1e-6
            )
            lrp_heat = cv2.applyColorMap(
                (relevance_map * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
            )
            lrp_res = cv2.resize(
                lrp_heat, img_res.shape[:2][::-1], interpolation=cv2.INTER_LINEAR
            )
            lrp_overlay = cv2.addWeighted(lrp_res, 0.4, img_res, 0.6, 0)

            lrp_dir = os.path.join("lrp_outputs", "transformer_rnd")
            os.makedirs(lrp_dir, exist_ok=True)
            cv2.imwrite(os.path.join(lrp_dir, f"lrp_step{step}.png"), lrp_overlay)

        if step % 50000 == 0 or step == num_steps - 1:
            ckpt_dir = os.path.join("checkpoints", "transformer_rnd")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(
                agent.policy.state_dict(),
                os.path.join(ckpt_dir, f"transformer_rnd_step{step}.pth"),
            )

    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="coinrun")
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--rollout_length", type=int, default=None)
    args = parser.parse_args()

    train_transformer_rnd(
        env_name=args.env,
        num_steps=args.steps,
        device=args.device,
        num_envs=args.num_envs,
        rollout_length=args.rollout_length,
    )
