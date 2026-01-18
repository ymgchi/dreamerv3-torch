"""
Evaluate trained DreamerV3 MilliSign v4/v5 model and save video
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
from collections import deque


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def to_attrdict(d):
    if isinstance(d, dict):
        return AttrDict({k: to_attrdict(v) for k, v in d.items()})
    return d


class MilliSignEnvV4Video:
    """MilliSign v4 environment for video recording."""

    HISTORY_LENGTH = 5

    def __init__(self, seed=42, size=(1280, 720), person_speed=1.0, target_distance=2.5):
        self._rng = np.random.RandomState(seed)
        self._size = size
        self._person_speed_mult = person_speed
        self._target_distance_init = target_distance

        self._env = HoverAviary(
            gui=False,
            obs=ObservationType.KIN,
            act=ActionType.VEL,
            pyb_freq=240,
            ctrl_freq=30,
        )

        self._episode_step = 0
        self._action_repeat = 1

        # Radar parameters
        self._radar_max_range = 15.0
        self._radar_min_range = 0.3
        self._radar_fov_azimuth = 120
        self._radar_fov_elevation = 40
        self._radar_noise_distance = 0.01
        self._radar_noise_angle = 1.0

        # Person/target parameters
        self._person_pos = np.array([3.0, 0.0, 0.0])
        self._person_vel = np.zeros(3)
        self._tag_height = 1.5
        self._target_distance = self._target_distance_init
        self._target_height = 1.8

        # Detection tracking
        self._detection_streak = 0
        self._lost_steps = 0

        # History buffers
        self._radar_history = deque(maxlen=self.HISTORY_LENGTH)
        self._drone_pos_history = deque(maxlen=self.HISTORY_LENGTH)

        # Previous positions
        self._prev_person_pos = None
        self._prev_drone_pos = None
        self._prev_action = None
        self._last_known_direction = np.array([1.0, 0.0, 0.0])

        # Curriculum stage (full speed for evaluation)
        self._current_stage = 5

        # Create visual markers
        self._create_markers()

    def _create_markers(self):
        """Create visual markers for person and tag."""
        # Person body - LARGE bright magenta cylinder
        self._person_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.6,
            length=2.0,
            rgbaColor=[1.0, 0.0, 1.0, 1.0],
            physicsClientId=self._env.CLIENT
        )
        self._person_marker = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self._person_visual,
            basePosition=[self._person_pos[0], self._person_pos[1], 1.0],
            physicsClientId=self._env.CLIENT
        )

        # Person head
        self._head_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.35,
            rgbaColor=[1.0, 0.8, 0.6, 1.0],
            physicsClientId=self._env.CLIENT
        )
        self._head_marker = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self._head_visual,
            basePosition=[self._person_pos[0], self._person_pos[1], 2.2],
            physicsClientId=self._env.CLIENT
        )

        # Tag - RED sphere on person
        self._tag_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.25,
            rgbaColor=[1.0, 0.0, 0.0, 1.0],
            physicsClientId=self._env.CLIENT
        )
        self._tag_marker = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self._tag_visual,
            basePosition=[self._person_pos[0], self._person_pos[1], self._tag_height],
            physicsClientId=self._env.CLIENT
        )

        # Target position - CYAN sphere
        self._target_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.3,
            rgbaColor=[0.0, 1.0, 1.0, 0.8],
            physicsClientId=self._env.CLIENT
        )
        self._target_marker = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self._target_visual,
            basePosition=self._get_target_position().tolist(),
            physicsClientId=self._env.CLIENT
        )

        # Ground plane marker
        self._ground_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[10, 10, 0.01],
            rgbaColor=[0.3, 0.5, 0.3, 1.0],
            physicsClientId=self._env.CLIENT
        )
        self._ground_marker = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self._ground_visual,
            basePosition=[0, 0, -0.01],
            physicsClientId=self._env.CLIENT
        )

    def _update_person_position(self):
        """Update person position with smooth random walk."""
        dt = 1.0 / 30.0
        max_turn_rate = 1.0  # radians per second
        
        # Set new target direction every 60 steps
        if self._episode_step % 60 == 0:
            self._target_angle = self._rng.uniform(0, 2 * np.pi)
            self._target_speed = self._rng.uniform(0.6, 1.2) * self._person_speed_mult
        
        # Initialize if needed
        if not hasattr(self, "_current_angle"):
            self._current_angle = 0.0
            self._target_angle = 0.0
            self._target_speed = 0.5 * self._person_speed_mult
            self._current_speed = 0.5 * self._person_speed_mult
        
        # Smoothly interpolate angle (shortest path)
        angle_diff = self._target_angle - self._current_angle
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        max_change = max_turn_rate * dt
        angle_change = np.clip(angle_diff, -max_change, max_change)
        self._current_angle += angle_change
        
        # Smoothly interpolate speed
        speed_diff = self._target_speed - self._current_speed
        self._current_speed += np.clip(speed_diff, -0.5 * dt, 0.5 * dt)
        
        # Update velocity and position
        self._person_vel = np.array([
            self._current_speed * np.cos(self._current_angle),
            self._current_speed * np.sin(self._current_angle),
            0.0
        ])
        self._person_pos += self._person_vel * dt
        self._person_pos = np.clip(self._person_pos, -4.0, 4.0)
        self._person_pos[2] = 0.0

        self._person_pos[2] = 0.0

        # Update markers
        p.resetBasePositionAndOrientation(
            self._person_marker,
            [self._person_pos[0], self._person_pos[1], 1.0],
            [0, 0, 0, 1],
            physicsClientId=self._env.CLIENT
        )
        p.resetBasePositionAndOrientation(
            self._head_marker,
            [self._person_pos[0], self._person_pos[1], 2.2],
            [0, 0, 0, 1],
            physicsClientId=self._env.CLIENT
        )
        p.resetBasePositionAndOrientation(
            self._tag_marker,
            [self._person_pos[0], self._person_pos[1], self._tag_height],
            [0, 0, 0, 1],
            physicsClientId=self._env.CLIENT
        )
        target_pos = self._get_target_position()
        p.resetBasePositionAndOrientation(
            self._target_marker,
            target_pos.tolist(),
            [0, 0, 0, 1],
            physicsClientId=self._env.CLIENT
        )

        return self._person_pos.copy()

    def _get_tag_position(self):
        tag_pos = self._person_pos.copy()
        tag_pos[2] = self._tag_height
        return tag_pos

    def _get_target_position(self, drone_pos=None):
        """Calculate target position - matches training environment logic.

        Target is 2.5m behind person in their movement direction.
        If person is stationary, target is on line from drone to person.
        """
        tag_pos = self._get_tag_position()
        person_speed = np.linalg.norm(self._person_vel[:2])

        if person_speed > 0.05:
            # Person is moving - target is behind them
            person_dir = self._person_vel[:2] / person_speed
            target_xy = tag_pos[:2] - self._target_distance * person_dir
        elif drone_pos is not None:
            # Person is stationary - target is on line from drone to person
            rel_pos = tag_pos[:2] - drone_pos[:2]
            dist_to_tag = np.linalg.norm(rel_pos)
            if dist_to_tag > 0.1:
                direction = rel_pos / dist_to_tag
                target_xy = tag_pos[:2] - self._target_distance * direction
            else:
                target_xy = tag_pos[:2] - self._target_distance * self._last_known_direction[:2]
        else:
            # Fallback to -X direction
            target_xy = tag_pos[:2] - np.array([self._target_distance, 0.0])

        target_pos = np.array([target_xy[0], target_xy[1], self._target_height], dtype=np.float32)
        return target_pos

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
            distance = np.clip(distance, 0, self._radar_max_range)
            signal_strength = 0.1

        return np.array([distance, azimuth, elevation, signal_strength, float(detected)], dtype=np.float32)

    def _get_history_observation(self):
        history = list(self._radar_history)
        while len(history) < self.HISTORY_LENGTH:
            history.insert(0, np.zeros(5, dtype=np.float32))
        return np.concatenate(history)

    def _get_drone_yaw(self, obs_flat):
        if len(obs_flat) >= 10:
            return obs_flat[9]
        return 0.0

    def _get_heading_to_target(self, drone_pos, tag_pos, drone_yaw):
        rel_pos = tag_pos[:2] - drone_pos[:2]
        target_angle = np.arctan2(rel_pos[1], rel_pos[0])
        heading_error = target_angle - drone_yaw
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        return heading_error, target_angle

    def render(self, drone_pos):
        """Render scene from chase camera behind drone."""
        import cv2
        width, height = self._size
        tag_pos_3d = [self._person_pos[0], self._person_pos[1], self._tag_height]

        # Chase camera - behind and above drone, looking toward person
        direction = np.array([tag_pos_3d[0] - drone_pos[0], tag_pos_3d[1] - drone_pos[1], 0])
        dist = np.linalg.norm(direction)
        if dist > 0.1:
            direction = direction / dist
        else:
            direction = np.array([1, 0, 0])

        cam_back = 3.0
        cam_height = 1.5
        cam_pos = [
            drone_pos[0] - direction[0] * cam_back,
            drone_pos[1] - direction[1] * cam_back,
            drone_pos[2] + cam_height
        ]

        # Look at midpoint between drone and target
        look_at = [
            (drone_pos[0] + tag_pos_3d[0]) / 2,
            (drone_pos[1] + tag_pos_3d[1]) / 2,
            (drone_pos[2] + tag_pos_3d[2]) / 2
        ]

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=look_at,
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

        frame = np.array(rgb, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        frame = frame.copy()  # Make writeable
        return frame

    @property
    def observation_space(self):
        import gym
        # v4 state: drone(72) + radar(5) + history(25) + vel(3) + pred_dir(2) + rel_vel(3) + target(3) + heading(2) + curriculum(1) + streak(1) + lost(1) = 118
        return gym.spaces.Dict({
            "state": gym.spaces.Box(-np.inf, np.inf, (118,), dtype=np.float32)
        })

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self):
        obs, _ = self._env.reset()
        self._episode_step = 0
        self._detection_streak = 0
        self._lost_steps = 0
        self._radar_history.clear()
        self._drone_pos_history.clear()
        self._prev_person_pos = None
        self._prev_drone_pos = None
        self._prev_action = None

        self._person_pos = np.array([
            self._rng.uniform(1.5, 3.5),
            self._rng.uniform(-1.5, 1.5),
            0.0
        ])
        self._person_vel = np.zeros(3)

        # Recreate markers after env reset (env reset clears pybullet objects)
        self._create_markers()

        obs_flat = np.array(obs, dtype=np.float32).flatten()
        drone_pos = obs_flat[:3]
        tag_pos = self._get_tag_position()
        radar_obs = self._simulate_radar_detection(drone_pos, tag_pos)
        self._radar_history.append(radar_obs.copy())

        drone_yaw = self._get_drone_yaw(obs_flat)
        heading_error, target_angle = self._get_heading_to_target(drone_pos, tag_pos, drone_yaw)

        extended_obs = np.concatenate([
            obs_flat,
            radar_obs,
            self._get_history_observation(),
            np.zeros(3, dtype=np.float32),  # person_vel
            np.zeros(2, dtype=np.float32),  # pred_direction
            np.zeros(3, dtype=np.float32),  # rel_vel
            np.array([self._target_distance, self._target_height, 0.0], dtype=np.float32),
            np.array([heading_error, target_angle], dtype=np.float32),
            np.array([float(self._current_stage)], dtype=np.float32),
            np.array([0.0], dtype=np.float32),  # streak
            np.array([0.0], dtype=np.float32),  # lost_steps
        ])

        return {"state": extended_obs.astype(np.float32), "is_first": True, "is_terminal": False}

    def step(self, action):
        if len(action.shape) == 1:
            action = action.reshape(self._env.action_space.shape)

        reward = 0.0
        obs, _, terminated, truncated, _ = self._env.step(action)
        self._episode_step += 1

        obs_flat = np.array(obs, dtype=np.float32).flatten()
        drone_pos = obs_flat[:3]
        self._update_person_position()
        tag_pos = self._get_tag_position()
        radar_obs = self._simulate_radar_detection(drone_pos, tag_pos)
        self._radar_history.append(radar_obs.copy())

        detected = radar_obs[4] > 0.5
        if detected:
            self._detection_streak += 1
            self._lost_steps = 0
        else:
            self._detection_streak = 0
            self._lost_steps += 1

        # V15 reward function
        target_pos = self._get_target_position()
        distance = np.linalg.norm(drone_pos - target_pos)
        
        # 1. Distance reward (main)
        reward = 1.0 - (distance / 2.0)
        reward = np.clip(reward, -1.0, 1.0)
        
        # 2. Detection bonus
        if detected:
            reward += 0.3
        
        # 3. Success bonus (sparse)
        if distance < 0.5 and detected:
            reward += 0.5

        # Person velocity estimate
        if self._prev_person_pos is not None:
            person_vel_est = (self._person_pos - self._prev_person_pos) * 30.0
        else:
            person_vel_est = np.zeros(3)

        # Predicted direction
        if np.linalg.norm(person_vel_est[:2]) > 0.05:
            pred_dir = person_vel_est[:2] / np.linalg.norm(person_vel_est[:2])
        else:
            pred_dir = np.zeros(2)

        # Relative velocity
        if self._prev_drone_pos is not None:
            drone_vel = (drone_pos - self._prev_drone_pos) * 30.0
            rel_vel = drone_vel - person_vel_est
        else:
            rel_vel = np.zeros(3)

        drone_yaw = self._get_drone_yaw(obs_flat)
        heading_error, target_angle = self._get_heading_to_target(drone_pos, tag_pos, drone_yaw)

        distance_error = radar_obs[0] - self._target_distance if detected else 0.0

        extended_obs = np.concatenate([
            obs_flat,
            radar_obs,
            self._get_history_observation(),
            person_vel_est.astype(np.float32),
            pred_dir.astype(np.float32),
            rel_vel.astype(np.float32),
            np.array([self._target_distance, self._target_height, distance_error], dtype=np.float32),
            np.array([heading_error, target_angle], dtype=np.float32),
            np.array([float(self._current_stage)], dtype=np.float32),
            np.array([min(self._detection_streak / 30.0, 1.0)], dtype=np.float32),
            np.array([min(self._lost_steps / 30.0, 1.0)], dtype=np.float32),
        ])

        self._prev_drone_pos = drone_pos.copy()
        self._prev_person_pos = self._person_pos.copy()
        self._prev_action = action.flatten().copy()

        target_pos = self._get_target_position(drone_pos)
        position_error = np.linalg.norm(drone_pos - target_pos)
        info = {
            "drone_pos": drone_pos,
            "tag_pos": tag_pos,
            "distance": np.linalg.norm(drone_pos - tag_pos),
            "position_error": position_error,  # Distance to TARGET, not person
            "detected": detected,
            "target_pos": target_pos,
        }

        return {"state": extended_obs.astype(np.float32), "is_first": False, "is_terminal": terminated}, reward, terminated or truncated, info

    def close(self):
        self._env.close()


def add_overlay(frame, ep, step, reward, dist, detected, target_dist=2.5, drone_pos=None, tag_pos=None):
    """Add text overlay to frame."""
    try:
        import cv2
        frame = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        bg_color = (0, 0, 0)

        detected_color = (0, 255, 0) if detected else (0, 0, 255)
        dist_color = (0, 255, 0) if abs(dist - target_dist) < 1.0 else (255, 165, 0) if abs(dist - target_dist) < 2.0 else (0, 0, 255)

        texts = [
            (f"MilliSign v4/v5 Drone Follow - Episode {ep}", color, 1.0),
            (f"Step: {step}/242  Reward: {reward:.1f}", color, 0.8),
            (f"Distance to Target: {dist:.2f}m (goal: {target_dist}m)", dist_color, 0.8),
            (f"Radar Status: {'TRACKING' if detected else 'LOST TARGET'}", detected_color, 0.8),
        ]

        if drone_pos is not None:
            texts.append((f"Drone: ({drone_pos[0]:.1f}, {drone_pos[1]:.1f}, {drone_pos[2]:.1f})", color, 0.6))
        if tag_pos is not None:
            texts.append((f"Person: ({tag_pos[0]:.1f}, {tag_pos[1]:.1f})", color, 0.6))

        y = 40
        for text, text_color, scale in texts:
            (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
            cv2.rectangle(frame, (8, y - th - 5), (16 + tw, y + 8), bg_color, -1)
            cv2.putText(frame, text, (12, y), font, scale, text_color, 2)
            y += int(40 * scale) + 10

        legend_y = frame.shape[0] - 30
        cv2.putText(frame, "Magenta=Person  Red=Tag  Cyan=Target",
                    (10, legend_y), font, 0.6, (200, 200, 200), 1)

    except ImportError:
        pass
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="/workspace/logdir/millisignv5")
    parser.add_argument("--config", type=str, default="millisignv5")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output", type=str, default="millisignv5_eval.mp4")
    parser.add_argument("--person-speed", type=float, default=1.0)
    parser.add_argument("--target-distance", type=float, default=2.5)
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
    if args.config in configs:
        recursive_update(config, configs[args.config])
    config = to_attrdict(config)
    config.device = 'cpu'
    config.num_actions = 4

    print(f"Creating environment (person_speed={args.person_speed}, target_distance={args.target_distance})...")
    env = MilliSignEnvV4Video(seed=42, size=(1280, 720), person_speed=args.person_speed, target_distance=args.target_distance)

    # Load checkpoint
    checkpoint_path = logdir / "latest.pt"
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)

    print(f"Checkpoint step: {checkpoint.get('step', 'unknown')}")

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

            frame = env.render(info["drone_pos"])
            frame = add_overlay(frame, ep + 1, step, episode_reward, info["position_error"], info["detected"],
                               target_dist=2.5, drone_pos=info["drone_pos"], tag_pos=info["tag_pos"])
            frames.append(frame)

            if step % 30 == 0:
                print(f"  Step {step}: Reward={episode_reward:.1f}, PosErr={info['position_error']:.2f}m, Detected={info['detected']}")

        total_rewards.append(episode_reward)
        print(f"  Final: Reward={episode_reward:.1f}, Steps={step}")

    env.close()

    # Save video
    output_path = pathlib.Path(args.output)
    print(f"\nSaving video to {output_path}")
    imageio.mimsave(str(output_path), frames, fps=30)

    print(f"\nEvaluation complete!")
    print(f"Average reward: {np.mean(total_rewards):.1f}")
    print(f"Video saved: {output_path}")


if __name__ == "__main__":
    main()
