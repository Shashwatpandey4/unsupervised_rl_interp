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

from agents.icm import ICM_Agent, preprocess_observation
from analysis.gradcam import GradCAM
from procgen import ProcgenEnv
from utils.logger import Logger


def train_icm(
    env_name="coinrun",
    num_steps=100000,
    device="cuda",
    num_envs=8,
    **agent_kwargs,
):
    # Create Procgen environment
    env = ProcgenEnv(num_envs=num_envs, env_name=env_name, render_mode="rgb_array")

    # Instantiate agent
    agent = ICM_Agent(
        env, device=torch.device(device), num_envs=num_envs, **agent_kwargs
    )

    # Logger and interpretability
    logger = Logger(agent_name="icm")
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
            "lr": agent.q_optimizer.param_groups[0]["lr"],
            "batch_size": agent.batch_size,
            "target_update_freq": agent.target_update_freq,
            "update_interval": agent.update_interval,
            "beta": agent.beta,
        }
    )

    # Initialize buffers and counters
    obs = env.reset()  # dict with key "rgb"
    episode_rewards = np.zeros(num_envs)
    episode_intrinsic = np.zeros(num_envs)
    episode_extrinsic = np.zeros(num_envs)
    reward_queue = deque(maxlen=100)

    pbar = trange(num_steps, desc="Training ICM", dynamic_ncols=True)
    for step in pbar:
        # Select actions
        actions = [agent.select_action(o) for o in obs["rgb"]]
        next_obs, rewards, dones, infos = env.step(np.array(actions))

        # Compute intrinsic rewards per env
        intrinsic_rewards = []
        for i in range(num_envs):
            obs_t = preprocess_observation(obs["rgb"][i]).unsqueeze(0).to(agent.device)
            next_t = (
                preprocess_observation(next_obs["rgb"][i]).unsqueeze(0).to(agent.device)
            )
            onehot = torch.zeros(1, env.action_space.n, device=agent.device)
            onehot[0, actions[i]] = 1
            with torch.no_grad():
                _, pred_phi, true_phi = agent.icm(obs_t, next_t, onehot)
            intrinsic = 0.5 * (pred_phi - true_phi.detach()).pow(2).sum().item()
            intrinsic_rewards.append(intrinsic)

        beta = agent.beta

        # Add transitions to replay buffer & log
        for i in range(num_envs):
            total_r = rewards[i] + beta * intrinsic_rewards[i]
            agent.replay_buffer.add(
                (obs["rgb"][i], actions[i], total_r, next_obs["rgb"][i], dones[i])
            )
            episode_rewards[i] += total_r
            episode_intrinsic[i] += intrinsic_rewards[i]
            episode_extrinsic[i] += rewards[i]

            logger.log_scalar("intrinsic_reward", intrinsic_rewards[i], step)
            logger.log_scalar("extrinsic_reward", rewards[i], step)
            logger.log_scalar("total_reward", total_r, step)

        obs = next_obs

        # Update networks periodically
        if (
            len(agent.replay_buffer) >= agent.batch_size
            and step % agent.update_interval == 0
        ):
            q_loss, icm_loss = agent.update()
            logger.log_scalar("q_loss", q_loss, step)
            logger.log_scalar("icm_loss", icm_loss, step)

        if step % agent.target_update_freq == 0:
            agent.target_q_net.load_state_dict(agent.q_net.state_dict())

        # Periodic flushing of logs
        if step % 1000 == 0:
            logger.flush()

        # Episode end logging
        for i in range(num_envs):
            if dones[i]:
                reward_queue.append(episode_rewards[i])
                avg = sum(reward_queue) / len(reward_queue)
                logger.log_scalar("episode_reward", episode_rewards[i], step)
                logger.log_scalar(
                    "episode_intrinsic_reward", episode_intrinsic[i], step
                )
                logger.log_scalar(
                    "episode_extrinsic_reward", episode_extrinsic[i], step
                )
                pbar.set_postfix(avg_reward=f"{avg:.2f}")
                episode_rewards[i] = 0
                episode_intrinsic[i] = 0
                episode_extrinsic[i] = 0

        # Save model checkpoint periodically
        if step % 50000 == 0 or step == num_steps - 1:
            checkpoint_dir = os.path.join("checkpoints", "icm")
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(
                agent.q_net.state_dict(),
                os.path.join(checkpoint_dir, f"q_net_step{step}.pth"),
            )
            torch.save(
                agent.icm.state_dict(),
                os.path.join(checkpoint_dir, f"icm_step{step}.pth"),
            )

        # Save Grad-CAM visuals safely
        if step % 5000 == 0:
            try:
                frame = (
                    preprocess_observation(obs["rgb"][0]).unsqueeze(0).to(agent.device)
                )
                cam = gradcam.generate_cam(frame)
                img = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype(np.uint8)
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

                cam_dir = os.path.join("cam_outputs", "icm")
                os.makedirs(cam_dir, exist_ok=True)
                cv2.imwrite(os.path.join(cam_dir, f"cam_step{step}.png"), overlay)

                frame_dir = os.path.join("frames", "icm")
                os.makedirs(frame_dir, exist_ok=True)
                cv2.imwrite(os.path.join(frame_dir, f"frame_step{step}.png"), img_res)
            except Exception as e:
                print(f"[Grad-CAM ERROR at step {step}] {e}")

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
            lrp_dir = os.path.join("lrp_outputs", "icm")
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

    train_icm(
        env_name=args.env,
        num_steps=args.steps,
        device=args.device,
        num_envs=args.num_envs,
    )
