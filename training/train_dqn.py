# training/train_dqn.py

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

from agents.dqn_agent import DQNAgent, preprocess_observation
from analysis.gradcam import GradCAM
from procgen import ProcgenEnv
from utils.logger import Logger


def train_dqn(
    env_name="coinrun",
    num_steps=100000,
    device="cuda",
    num_envs=8,
    **agent_kwargs,
):
    # Create Procgen environment
    env = ProcgenEnv(num_envs=num_envs, env_name=env_name, render_mode="rgb_array")

    # Instantiate agent
    agent = DQNAgent(
        env, device=torch.device(device), num_envs=num_envs, **agent_kwargs
    )

    # Logger and interpretability
    logger = Logger(agent_name="dqn")
    print("Available layers in q_net:")
    print([name for name, _ in agent.q_net.named_modules()])
    gradcam = GradCAM(agent.q_net, target_layer_name="conv.4")

    # Log config
    logger.log_config(
        {
            "env": env_name,
            "device": device,
            "num_steps": num_steps,
            "num_envs": num_envs,
            "gamma": agent.gamma,
            "lr": agent.optimizer.param_groups[0]["lr"],
            "batch_size": agent.batch_size,
            "update_interval": agent.update_interval,
            "target_update_freq": agent.target_update_freq,
        }
    )

    obs = env.reset()
    episode_rewards = np.zeros(num_envs)
    reward_queue = deque(maxlen=100)

    pbar = trange(num_steps, desc="Training DQN", dynamic_ncols=True)
    for step in pbar:
        # 1) select actions
        actions = np.array([agent.select_action(o) for o in obs["rgb"]])
        next_obs, rewards, dones, infos = env.step(actions)

        # 2) add transitions & log extrinsic rewards
        for i in range(num_envs):
            agent.replay_buffer.add(
                (obs["rgb"][i], actions[i], rewards[i], next_obs["rgb"][i], dones[i])
            )
            episode_rewards[i] += rewards[i]
            logger.log_scalar("extrinsic_reward", rewards[i], step)

        obs = next_obs

        # 3) update networks periodically
        if (
            len(agent.replay_buffer) >= agent.batch_size
            and step % agent.update_interval == 0
        ):
            q_loss = agent.update()
            logger.log_scalar("q_loss", q_loss, step)

        if step % agent.target_update_freq == 0:
            agent.target_q_net.load_state_dict(agent.q_net.state_dict())

        # Periodic flushing of logs
        if step % 1000 == 0:
            logger.flush()

        # 4) episode-end logging
        for i in range(num_envs):
            if dones[i]:
                reward_queue.append(episode_rewards[i])
                avg = sum(reward_queue) / len(reward_queue)
                logger.log_scalar("episode_reward", episode_rewards[i], step)
                pbar.set_postfix(avg_reward=f"{avg:.2f}")
                episode_rewards[i] = 0

        # Save model checkpoint periodically
        if step % 50000 == 0 or step == num_steps - 1:
            checkpoint_dir = os.path.join("checkpoints", "dqn")
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(
                agent.q_net.state_dict(),
                os.path.join(checkpoint_dir, f"q_net_step{step}.pth"),
            )

        # 5) save Grad-CAM & frames
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
                heat,
                (img_res.shape[1], img_res.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            overlay = cv2.addWeighted(heat_res, 0.4, img_res, 0.6, 0)

            cam_dir = os.path.join("cam_outputs", "dqn")
            os.makedirs(cam_dir, exist_ok=True)
            cv2.imwrite(os.path.join(cam_dir, f"cam_step{step}.png"), overlay)

            frame_dir = os.path.join("frames", "dqn")
            os.makedirs(frame_dir, exist_ok=True)
            cv2.imwrite(os.path.join(frame_dir, f"frame_step{step}.png"), img_res)
            # ---- LRP START ----
            from analysis.lrp import generate_lrp_map

            # Generate LRP map
            relevance_map = generate_lrp_map(
                agent.q_net, frame.cpu()
            )  # make sure frame is on CPU

            # Normalize and convert to heatmap
            relevance_map = relevance_map.sum(axis=0)
            relevance_map = (relevance_map - relevance_map.min()) / (
                relevance_map.max() - relevance_map.min() + 1e-6
            )
            lrp_heat = cv2.applyColorMap(
                (relevance_map * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
            )

            # Resize to match original
            lrp_res = cv2.resize(
                lrp_heat,
                (img_res.shape[1], img_res.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            lrp_overlay = cv2.addWeighted(lrp_res, 0.4, img_res, 0.6, 0)

            # Save LRP result
            lrp_dir = os.path.join("lrp_outputs", "dqn")
            os.makedirs(lrp_dir, exist_ok=True)
            cv2.imwrite(os.path.join(lrp_dir, f"lrp_step{step}.png"), lrp_overlay)
            # ---- LRP END ----
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="coinrun")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_envs", type=int, default=8)
    args = parser.parse_args()

    train_dqn(
        env_name=args.env,
        num_steps=args.steps,
        device=args.device,
        num_envs=args.num_envs,
    )
