import argparse
import os
import sys

import yaml

# Allow importing from training/
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from training.train_dqn import train_dqn
from training.train_icm import train_icm
from training.train_ppo import train_ppo
from training.train_rnd import train_rnd
from training.train_transformer import train_transformer_rnd


def load_config(agent_name, config_path=None):
    if config_path is None:
        config_path = os.path.join("configs", f"{agent_name}.yaml")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=["dqn", "rnd", "icm", "ppo", "transformer_rnd"],
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Optional path to YAML config file."
    )
    args = parser.parse_args()

    print(f"\nLaunching {args.agent.upper()} agent...")

    config = load_config(args.agent, args.config)
    env_name = config.get("env", "coinrun")
    num_steps = config.get("steps", 100000)
    device = config.get("device", "cuda")
    agent_cfg = config.get("agent", {})

    if args.agent == "dqn":
        train_dqn(env_name, num_steps, device, **agent_cfg)
    elif args.agent == "rnd":
        train_rnd(env_name, num_steps, device, **agent_cfg)
    elif args.agent == "icm":
        train_icm(env_name, num_steps, device, **agent_cfg)
    elif args.agent == "ppo":
        train_ppo(env_name, num_steps, device, **agent_cfg)
    elif args.agent == "transformer_rnd":
        train_transformer_rnd(env_name, num_steps, device, **agent_cfg)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")
