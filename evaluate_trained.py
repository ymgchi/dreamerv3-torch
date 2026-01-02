"""
Evaluate trained DreamerV3 drone model and save video
"""
import argparse
import functools
import pathlib
import sys
import numpy as np
import torch
import imageio

sys.path.insert(0, "/workspace/gym-pybullet-drones")
sys.path.append(str(pathlib.Path(__file__).parent))

import ruamel.yaml as yaml
import tools
import models
import envs.drone_trajectory as drone_trajectory
import envs.wrappers as wrappers


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def to_attrdict(d):
    """Recursively convert dict to AttrDict"""
    if isinstance(d, dict):
        return AttrDict({k: to_attrdict(v) for k, v in d.items()})
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--task", type=str, default="circle")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output", type=str, default="trained_eval.mp4")
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir)

    # Load config
    configs = yaml.safe_load(
        (pathlib.Path(__file__).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    config = {}
    recursive_update(config, configs["defaults"])
    recursive_update(config, configs["trajectory_circle"])
    config = to_attrdict(config)
    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.num_actions = 4  # Drone has 4 motor controls

    # Create environment
    print("Creating environment...")
    env = drone_trajectory.PyBulletDroneTrajectory(
        task=args.task,
        action_repeat=config.action_repeat,
        size=(256, 256),
        seed=42,
    )
    env = wrappers.NormalizeActions(env)
    env = wrappers.TimeLimit(env, config.time_limit)

    # Load checkpoint
    checkpoint_path = logdir / "latest.pt"
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.device)

    # Create agent
    print("Creating agent...")
    obs_space = env.observation_space
    act_space = env.action_space

    step = 0

    # Create world model
    wm = models.WorldModel(obs_space, act_space, step, config)
    task_behavior = models.ImagBehavior(config, wm)

    # Load weights
    agent_state = checkpoint["agent_state_dict"]

    # Filter and fix state dict for world model (handle torch.compile prefix)
    wm_state = {}
    for k, v in agent_state.items():
        if k.startswith("_wm."):
            new_key = k.replace("_wm.", "")
            # Remove _orig_mod. prefix from torch.compile
            new_key = new_key.replace("_orig_mod.", "")
            wm_state[new_key] = v
    wm.load_state_dict(wm_state)

    # Filter state dict for task behavior
    tb_state = {}
    for k, v in agent_state.items():
        if k.startswith("_task_behavior."):
            new_key = k.replace("_task_behavior.", "")
            new_key = new_key.replace("_orig_mod.", "")
            tb_state[new_key] = v
    task_behavior.load_state_dict(tb_state)

    wm.to(config.device)
    task_behavior.to(config.device)
    wm.eval()
    task_behavior.eval()

    print("Running evaluation...")
    frames = []
    total_rewards = []

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        step = 0

        # Initialize state
        latent = None
        action = None

        print(f"Episode {ep + 1}/{args.episodes}")

        while not done:
            # Preprocess observation
            obs_proc = {k: torch.tensor(v, device=config.device).unsqueeze(0).float()
                       for k, v in obs.items() if k in ["state", "image"]}
            obs_proc["is_first"] = torch.tensor([[obs.get("is_first", step == 0)]],
                                                 device=config.device, dtype=torch.bool)
            obs_proc["is_terminal"] = torch.tensor([[obs.get("is_terminal", False)]],
                                                    device=config.device, dtype=torch.bool)

            with torch.no_grad():
                # Encode observation
                obs_proc = wm.preprocess(obs_proc)
                embed = wm.encoder(obs_proc)

                # Update latent state
                latent, _ = wm.dynamics.obs_step(latent, action, embed, obs_proc["is_first"])

                # Get features and action (use stoch directly for simplicity)
                feat = wm.dynamics.get_feat(latent)
                actor = task_behavior.actor(feat)
                action = actor.mode()

            # Execute action
            action_np = action.squeeze(0).cpu().numpy()
            obs, reward, done, info = env.step(action_np)
            episode_reward += reward
            step += 1

            # Capture frame with overlay
            frame = obs["image"]
            frame = add_text_overlay(frame, ep + 1, step, episode_reward,
                                    info.get("dist_to_target", 0),
                                    info.get("target_pos", [0, 0, 0]))
            frames.append(frame)

            if step >= 150:
                break

        total_rewards.append(episode_reward)
        print(f"  Reward: {episode_reward:.1f}, Steps: {step}")

    env.close()

    # Save video
    output_path = logdir / args.output
    print(f"Saving video to {output_path}")
    imageio.mimsave(str(output_path), frames, fps=30)

    # Also copy to dreamerv3-torch directory
    import shutil
    dest_path = pathlib.Path(__file__).parent / args.output
    shutil.copy(output_path, dest_path)

    print(f"\nEvaluation complete!")
    print(f"Average reward: {np.mean(total_rewards):.1f}")
    print(f"Video saved: {dest_path}")


def add_text_overlay(frame, episode, step, reward, dist, target_pos):
    """Add text overlay to frame"""
    try:
        import cv2
        frame = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        cv2.putText(frame, f"Ep:{episode} Step:{step}", (10, 25), font, 0.5, color, 1)
        cv2.putText(frame, f"Reward:{reward:.1f}", (10, 45), font, 0.5, color, 1)
        cv2.putText(frame, f"Dist:{dist:.2f}m", (10, 65), font, 0.5, color, 1)
        cv2.putText(frame, f"Target:({target_pos[0]:.1f},{target_pos[1]:.1f},{target_pos[2]:.1f})",
                    (10, 85), font, 0.4, color, 1)
    except ImportError:
        pass
    return frame


if __name__ == "__main__":
    main()
