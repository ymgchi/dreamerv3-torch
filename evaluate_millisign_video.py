"""
Evaluate trained DreamerV3 MilliSign model and save video
"""
import argparse
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
import pybullet as p

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def to_attrdict(d):
    if isinstance(d, dict):
        return AttrDict({k: to_attrdict(v) for k, v in d.items()})
    return d


class MilliSignEnvVideo:
    """MilliSign environment for video recording."""

    def __init__(self, seed=42, size=(640, 480)):
        self._rng = np.random.RandomState(seed)
        self._size = size

        # Create headless environment
        self._env = HoverAviary(
            gui=False,
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            pyb_freq=240,
            ctrl_freq=30,
        )

        self._episode_step = 0
        self._action_repeat = 2

        # Radar parameters
        self._radar_max_range = 15.0
        self._radar_min_range = 0.5
        self._radar_fov_azimuth = 60
        self._radar_fov_elevation = 60
        self._radar_noise_distance = 0.01
        self._radar_noise_angle = 1.0

        # Person/target parameters
        self._person_pos = np.array([3.0, 0.0, 0.0])
        self._person_vel = np.zeros(3)
        self._tag_height = 1.5
        self._target_distance = 3.0
        self._target_height = 2.0

        # Create visual markers
        self._create_markers()

    def _create_markers(self):
        """Create visual markers for person and tag."""
        # Person body (cylinder)
        self._person_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.3,
            length=1.7,
            rgbaColor=[0.2, 0.6, 0.2, 0.8],
            physicsClientId=self._env.CLIENT
        )
        self._person_body = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self._person_visual,
            basePosition=[3.0, 0.0, 0.85],
            physicsClientId=self._env.CLIENT
        )

        # MilliSign tag (red box at chest)
        self._tag_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.1, 0.1, 0.02],
            rgbaColor=[1.0, 0.0, 0.0, 1.0],
            physicsClientId=self._env.CLIENT
        )
        self._tag_body = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self._tag_visual,
            basePosition=[3.0, 0.0, 1.5],
            physicsClientId=self._env.CLIENT
        )

    def _update_person_position(self):
        """Update person position with random walk."""
        if self._episode_step % 60 == 0:
            angle = self._rng.uniform(0, 2 * np.pi)
            speed = self._rng.uniform(0.3, 0.8)
            self._person_vel = np.array([
                speed * np.cos(angle),
                speed * np.sin(angle),
                0.0
            ])

        dt = 1.0 / 30.0
        self._person_pos += self._person_vel * dt
        self._person_pos = np.clip(self._person_pos, -5.0, 5.0)
        self._person_pos[2] = 0.0

        # Update visual markers
        p.resetBasePositionAndOrientation(
            self._person_body,
            [self._person_pos[0], self._person_pos[1], 0.85],
            [0, 0, 0, 1],
            physicsClientId=self._env.CLIENT
        )
        p.resetBasePositionAndOrientation(
            self._tag_body,
            [self._person_pos[0], self._person_pos[1], self._tag_height],
            [0, 0, 0, 1],
            physicsClientId=self._env.CLIENT
        )

        return self._person_pos.copy()

    def _get_tag_position(self):
        person_pos = self._update_person_position()
        tag_pos = person_pos.copy()
        tag_pos[2] = self._tag_height
        return tag_pos

    def _simulate_radar_detection(self, drone_pos, tag_pos):
        rel_pos = tag_pos - drone_pos
        distance = np.linalg.norm(rel_pos)
        azimuth = np.arctan2(rel_pos[1], rel_pos[0])
        elevation = np.arctan2(rel_pos[2], np.sqrt(rel_pos[0]**2 + rel_pos[1]**2))

        detected = True
        if distance > self._radar_max_range or distance < self._radar_min_range:
            detected = False
        if np.abs(np.degrees(azimuth)) > self._radar_fov_azimuth / 2:
            detected = False
        if np.abs(np.degrees(elevation)) > self._radar_fov_elevation / 2:
            detected = False

        if detected:
            distance += self._rng.normal(0, self._radar_noise_distance)
            azimuth += np.radians(self._rng.normal(0, self._radar_noise_angle))
            elevation += np.radians(self._rng.normal(0, self._radar_noise_angle))
            signal_strength = np.clip(1.0 / (distance / 5.0) ** 2, 0, 1)
        else:
            distance = azimuth = elevation = signal_strength = 0.0

        return np.array([distance, azimuth, elevation, signal_strength, float(detected)], dtype=np.float32)

    def render(self, drone_pos):
        """Render scene from chase camera."""
        width, height = self._size
        tag_pos = self._person_pos.copy()
        tag_pos[2] = self._tag_height

        # Chase camera behind and above drone
        cam_dist = 3.0
        cam_height = 1.5
        cam_pos = [
            drone_pos[0] - cam_dist * 0.7,
            drone_pos[1] - cam_dist * 0.7,
            drone_pos[2] + cam_height
        ]

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=drone_pos,
            cameraUpVector=[0, 0, 1],
            physicsClientId=self._env.CLIENT
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width / height,
            nearVal=0.1,
            farVal=100.0,
            physicsClientId=self._env.CLIENT
        )

        _, _, rgb, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._env.CLIENT
        )

        rgb_array = np.array(rgb, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        return rgb_array

    @property
    def observation_space(self):
        import gym
        return gym.spaces.Dict({
            "state": gym.spaces.Box(-np.inf, np.inf, (83,), dtype=np.float32)
        })

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self):
        obs, _ = self._env.reset()
        self._episode_step = 0

        self._person_pos = np.array([
            self._rng.uniform(2.0, 4.0),
            self._rng.uniform(-1.0, 1.0),
            0.0
        ])
        self._person_vel = np.zeros(3)

        obs_flat = np.array(obs, dtype=np.float32).flatten()
        drone_pos = obs_flat[:3]
        tag_pos = self._get_tag_position()
        radar_obs = self._simulate_radar_detection(drone_pos, tag_pos)

        extended_obs = np.concatenate([
            obs_flat,
            radar_obs,
            np.zeros(3, dtype=np.float32),
            np.array([self._target_distance, self._target_height, 0.0], dtype=np.float32),
        ])

        return {"state": extended_obs.astype(np.float32), "is_first": True, "is_terminal": False}

    def step(self, action):
        if len(action.shape) == 1:
            action = action.reshape(self._env.action_space.shape)

        reward = 0.0
        for _ in range(self._action_repeat):
            obs, _, terminated, truncated, _ = self._env.step(action)
            self._episode_step += 1

            obs_flat = np.array(obs, dtype=np.float32).flatten()
            drone_pos = obs_flat[:3]
            tag_pos = self._get_tag_position()
            radar_obs = self._simulate_radar_detection(drone_pos, tag_pos)

            detected = radar_obs[4] > 0.5
            if not detected:
                reward -= 2.0
            else:
                dist_err = np.abs(radar_obs[0] - self._target_distance)
                reward += max(0, 2.0 - dist_err) if dist_err >= 0.5 else 2.0
                height_err = np.abs(drone_pos[2] - self._target_height)
                reward += max(0, 1.0 - height_err)

            if terminated or truncated:
                break

        extended_obs = np.concatenate([
            obs_flat,
            radar_obs,
            self._person_vel.astype(np.float32),
            np.array([self._target_distance, self._target_height,
                     radar_obs[0] - self._target_distance if radar_obs[4] > 0.5 else 0.0], dtype=np.float32),
        ])

        info = {
            "drone_pos": drone_pos,
            "tag_pos": tag_pos,
            "distance": np.linalg.norm(drone_pos - tag_pos),
            "detected": detected,
        }

        return {"state": extended_obs.astype(np.float32), "is_first": False, "is_terminal": terminated}, reward, terminated or truncated, info

    def close(self):
        self._env.close()


def add_overlay(frame, ep, step, reward, dist, detected):
    """Add text overlay to frame."""
    try:
        import cv2
        frame = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        bg_color = (0, 0, 0)

        texts = [
            f"Episode: {ep}  Step: {step}",
            f"Reward: {reward:.1f}",
            f"Distance: {dist:.2f}m (target: 3.0m)",
            f"Radar: {'DETECTED' if detected else 'LOST'}",
        ]

        y = 30
        for text in texts:
            # Background
            (tw, th), _ = cv2.getTextSize(text, font, 0.6, 2)
            cv2.rectangle(frame, (8, y - th - 5), (12 + tw, y + 5), bg_color, -1)
            cv2.putText(frame, text, (10, y), font, 0.6, color, 2)
            y += 30

    except ImportError:
        pass
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="/workspace/logdir/millisign_follow")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output", type=str, default="millisign_eval.mp4")
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
    recursive_update(config, configs["millisign_follow"])
    config = to_attrdict(config)
    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.num_actions = 4

    print("Creating environment...")
    env = MilliSignEnvVideo(seed=42, size=(640, 480))

    # Load checkpoint
    checkpoint_path = logdir / "latest.pt"
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)

    # Create model
    print("Creating model...")
    obs_space = env.observation_space
    act_space = env.action_space

    wm = models.WorldModel(obs_space, act_space, 0, config)
    task_behavior = models.ImagBehavior(config, wm)

    # Load weights
    agent_state = checkpoint["agent_state_dict"]

    wm_state = {}
    for k, v in agent_state.items():
        if k.startswith("_wm."):
            new_key = k.replace("_wm.", "").replace("_orig_mod.", "")
            wm_state[new_key] = v
    wm.load_state_dict(wm_state)

    tb_state = {}
    for k, v in agent_state.items():
        if k.startswith("_task_behavior."):
            new_key = k.replace("_task_behavior.", "").replace("_orig_mod.", "")
            tb_state[new_key] = v
    task_behavior.load_state_dict(tb_state)

    wm.to(config.device)
    task_behavior.to(config.device)
    wm.eval()
    task_behavior.eval()

    print("\nRunning evaluation...")
    frames = []
    total_rewards = []

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        step = 0
        latent = None
        action = None

        print(f"\nEpisode {ep + 1}/{args.episodes}")

        while not done and step < 300:
            # Preprocess
            obs_proc = {"state": torch.tensor(obs["state"], device=config.device).unsqueeze(0).float()}
            obs_proc["is_first"] = torch.tensor([[obs["is_first"]]], device=config.device, dtype=torch.bool)
            obs_proc["is_terminal"] = torch.tensor([[obs["is_terminal"]]], device=config.device, dtype=torch.bool)
            obs_proc["image"] = torch.zeros((1, 64, 64, 3), device=config.device, dtype=torch.float32)

            with torch.no_grad():
                obs_proc = wm.preprocess(obs_proc)
                embed = wm.encoder(obs_proc)
                latent, _ = wm.dynamics.obs_step(latent, action, embed, obs_proc["is_first"])
                feat = wm.dynamics.get_feat(latent)
                actor = task_behavior.actor(feat)
                action = actor.mode()

            action_np = action.squeeze(0).cpu().numpy()
            obs, reward, done, info = env.step(action_np)
            episode_reward += reward
            step += 1

            # Render and save frame
            frame = env.render(info["drone_pos"])
            frame = add_overlay(frame, ep + 1, step, episode_reward, info["distance"], info["detected"])
            frames.append(frame)

            if step % 30 == 0:
                print(f"  Step {step}: Reward={episode_reward:.1f}, Dist={info['distance']:.2f}m")

        total_rewards.append(episode_reward)
        print(f"  Final: Reward={episode_reward:.1f}, Steps={step}")

    env.close()

    # Save video
    output_path = pathlib.Path(__file__).parent / args.output
    print(f"\nSaving video to {output_path}")
    imageio.mimsave(str(output_path), frames, fps=30)

    print(f"\nEvaluation complete!")
    print(f"Average reward: {np.mean(total_rewards):.1f}")
    print(f"Video saved: {output_path}")


if __name__ == "__main__":
    main()
