"""
Visualize trained DreamerV3 drone policy - saves video
"""
import argparse
import functools
import pathlib
import sys

import numpy as np
import torch
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import models
import tools
from envs.drone import PyBulletDrone
import envs.wrappers as wrappers


class Dreamer(torch.nn.Module):
    """Simplified Dreamer for evaluation only."""

    def __init__(self, obs_space, act_space, config):
        super(Dreamer, self).__init__()
        self._config = config
        self._step = 0
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)

    def policy(self, obs, state):
        """Get action from observation."""
        if state is None:
            latent = action = None
        else:
            latent, action = state

        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])

        # Use mean for deterministic evaluation (if available)
        if "mean" in latent:
            latent["stoch"] = latent["mean"]

        feat = self._wm.dynamics.get_feat(latent)
        actor = self._task_behavior.actor(feat)
        action = actor.mode()

        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()

        state = (latent, action)
        return action, state


def make_env(size=(256, 256)):
    """Create drone environment for visualization."""
    env = PyBulletDrone(
        task="hover",
        action_repeat=1,
        size=size,
        seed=0,
    )
    env = wrappers.NormalizeActions(env)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="/workspace/logdir/drone")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output", type=str, default="/workspace/logdir/drone/drone_hover.mp4")
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir)
    checkpoint_path = logdir / "latest.pt"

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading checkpoint from {checkpoint_path}")

    # Load config
    config_path = pathlib.Path(__file__).parent / "configs.yaml"
    configs = yaml.safe_load(config_path.read_text())

    # Merge defaults and drone config
    config = configs["defaults"].copy()
    for key, value in configs["drone"].items():
        if isinstance(value, dict) and key in config:
            config[key].update(value)
        else:
            config[key] = value

    # Convert to namespace
    class Config:
        pass
    cfg = Config()
    for key, value in config.items():
        setattr(cfg, key, value)

    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.num_actions = 4

    # Create environment
    print("Creating environment...")
    env = make_env(size=(256, 256))

    obs_space = env.observation_space
    act_space = env.action_space

    # Create agent
    print("Creating agent...")
    agent = Dreamer(obs_space, act_space, cfg).to(cfg.device)

    # Load checkpoint
    print("Loading weights...")
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device)

    # Remove _orig_mod. prefix from all keys (from torch.compile)
    state_dict = {}
    for key, value in checkpoint["agent_state_dict"].items():
        new_key = key
        # Handle nested _orig_mod. prefixes
        while "._orig_mod." in new_key:
            new_key = new_key.replace("._orig_mod.", ".")
        if new_key.startswith("_orig_mod."):
            new_key = new_key[10:]
        state_dict[new_key] = value

    agent.load_state_dict(state_dict, strict=False)
    agent.eval()

    print(f"\nRunning {args.episodes} episodes...")

    all_frames = []

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        state = None
        frames = []

        print(f"Episode {ep + 1}/{args.episodes}")

        while not done and step_count < 120:
            # Capture frame
            frame = env.render()
            frames.append(frame)

            # Convert observation to tensor
            obs_tensor = {}
            for key, value in obs.items():
                if key in ["state", "image"]:
                    obs_tensor[key] = torch.tensor(
                        value, dtype=torch.float32, device=cfg.device
                    ).unsqueeze(0)
                elif key in ["is_first", "is_terminal"]:
                    obs_tensor[key] = torch.tensor(
                        [value], dtype=torch.bool, device=cfg.device
                    )

            with torch.no_grad():
                action, state = agent.policy(obs_tensor, state)

            # Convert action to numpy
            action_np = action.squeeze(0).cpu().numpy()

            # Step environment
            obs, reward, done, info = env.step(action_np)
            total_reward += reward
            step_count += 1

        print(f"  Return: {total_reward:.1f}, Steps: {step_count}, Frames: {len(frames)}")
        all_frames.extend(frames)

    env.close()

    # Save video
    print(f"\nSaving video to {args.output}...")
    try:
        import imageio
        output_path = pathlib.Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        writer = imageio.get_writer(args.output, fps=30)
        for frame in all_frames:
            writer.append_data(frame)
        writer.close()
        print(f"Video saved: {args.output}")
        print(f"Total frames: {len(all_frames)}")

    except Exception as e:
        print(f"Error saving video: {e}")
        try:
            gif_path = args.output.replace(".mp4", ".gif")
            print(f"Trying GIF instead: {gif_path}")
            import imageio
            imageio.mimsave(gif_path, all_frames[::2], fps=15)
            print(f"GIF saved: {gif_path}")
        except Exception as e2:
            print(f"Error saving GIF: {e2}")

    print("Done!")


if __name__ == "__main__":
    main()
