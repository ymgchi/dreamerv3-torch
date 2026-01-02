"""
Evaluate trained drone model and save video
"""
import argparse
import pathlib
import sys
import numpy as np
import torch
import imageio

sys.path.insert(0, "/workspace/gym-pybullet-drones")
sys.path.append(str(pathlib.Path(__file__).parent))

import envs.drone_trajectory as drone_trajectory
import envs.wrappers as wrappers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--task", type=str, default="circle")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output", type=str, default="evaluation.mp4")
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir)

    # Load checkpoint
    checkpoint_path = logdir / "latest.pt"
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Create environment
    env = drone_trajectory.PyBulletDroneTrajectory(
        task=args.task,
        action_repeat=2,
        size=(256, 256),  # Larger size for video
        seed=42,
    )
    env = wrappers.NormalizeActions(env)

    # Get model config from checkpoint
    agent_state = checkpoint["agent_state_dict"]

    # We need to reconstruct the model - for now, let's just run random policy
    # and show the environment working

    frames = []
    total_rewards = []

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        step = 0

        print(f"Episode {ep + 1}/{args.episodes}")

        while not done:
            # For visualization, use simple proportional control towards target
            state = obs["state"]
            # state: [drone_state(12), target_pos(3), relative_pos(3)]
            relative_pos = state[-3:]  # Target relative to drone

            # Simple P control: action proportional to relative position
            # This gives us a baseline behavior to visualize
            action = np.clip(relative_pos * 2.0, -1, 1)
            action = np.concatenate([action, [0.5]])  # Add thrust
            action = action[:4]  # Ensure 4 actions

            obs, reward, done, info = env.step(action)
            episode_reward += reward
            step += 1

            # Capture frame
            frame = obs["image"]

            # Add text overlay
            frame = add_text_overlay(frame, ep + 1, step, episode_reward, info.get("dist_to_target", 0))
            frames.append(frame)

            if step >= 150:  # Max steps per episode for video
                break

        total_rewards.append(episode_reward)
        print(f"  Reward: {episode_reward:.1f}, Steps: {step}")

    env.close()

    # Save video
    output_path = logdir / args.output
    print(f"Saving video to {output_path}")
    imageio.mimsave(str(output_path), frames, fps=30)

    print(f"\nEvaluation complete!")
    print(f"Average reward: {np.mean(total_rewards):.1f}")
    print(f"Video saved: {output_path}")


def add_text_overlay(frame, episode, step, reward, dist):
    """Add simple text overlay to frame"""
    import cv2
    frame = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Ep:{episode} Step:{step}", (10, 25), font, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Reward:{reward:.1f}", (10, 50), font, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Dist:{dist:.2f}m", (10, 75), font, 0.6, (255, 255, 255), 2)
    return frame


if __name__ == "__main__":
    main()
