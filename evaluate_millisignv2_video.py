"""
Evaluate trained DreamerV3 MilliSign v2 model and save video
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


class MilliSignEnvV2Video:
    """MilliSign v2 environment for video recording."""

    def __init__(self, seed=42, size=(1280, 720)):
        self._rng = np.random.RandomState(seed)
        self._size = size

        # Create headless environment with VEL action type (v2 uses VEL)
        self._env = HoverAviary(
            gui=False,
            obs=ObservationType.KIN,
            act=ActionType.VEL,  # v2 uses velocity control
            pyb_freq=240,
            ctrl_freq=30,
        )

        self._episode_step = 0
        self._action_repeat = 1  # v2 uses action_repeat=1

        # Radar parameters (AWR1843 specs - v2)
        self._radar_max_range = 15.0
        self._radar_min_range = 0.3
        self._radar_fov_azimuth = 120   # ±60°
        self._radar_fov_elevation = 40  # ±20°
        self._radar_noise_distance = 0.01
        self._radar_noise_angle = 1.0

        # Person/target parameters (v2)
        self._person_pos = np.array([3.0, 0.0, 0.0])
        self._person_vel = np.zeros(3)
        self._tag_height = 1.5
        self._target_distance = 2.5  # v2 uses 2.5
        self._target_height = 1.8    # v2 uses 1.8

        # Curriculum stage (full speed for evaluation)
        self._person_speed = 0.8
        self._current_stage = 3  # Full stage

        # Create visual markers
        self._create_markers()

    def _create_markers(self):
        """Create visual markers for person and tag."""
        # Person body - LARGE bright magenta cylinder (very visible)
        self._person_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.6,  # Much larger
            length=2.0,
            rgbaColor=[1.0, 0.0, 1.0, 1.0],  # Magenta - stands out
            physicsClientId=self._env.CLIENT
        )
        self._person_body = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self._person_visual,
            basePosition=[3.0, 0.0, 1.0],
            physicsClientId=self._env.CLIENT
        )

        # Person head - LARGE sphere
        self._head_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.35,  # Larger
            rgbaColor=[1.0, 0.8, 0.6, 1.0],  # Skin color
            physicsClientId=self._env.CLIENT
        )
        self._head_body = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self._head_visual,
            basePosition=[3.0, 0.0, 2.2],
            physicsClientId=self._env.CLIENT
        )

        # MilliSign tag - LARGE bright red box
        self._tag_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.3, 0.3, 0.05],  # Much larger
            rgbaColor=[1.0, 0.0, 0.0, 1.0],  # Bright red
            physicsClientId=self._env.CLIENT
        )
        self._tag_body = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self._tag_visual,
            basePosition=[3.0, 0.0, 1.5],
            physicsClientId=self._env.CLIENT
        )

        # Target position marker - LARGE cyan sphere
        self._target_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.4,  # Larger
            rgbaColor=[0.0, 1.0, 1.0, 0.8],  # Cyan
            physicsClientId=self._env.CLIENT
        )
        self._target_marker = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self._target_visual,
            basePosition=[0.0, 0.0, 2.0],
            physicsClientId=self._env.CLIENT
        )

        # Drone marker - LARGE yellow sphere
        self._drone_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.4,  # Larger
            rgbaColor=[1.0, 1.0, 0.0, 1.0],  # Bright yellow
            physicsClientId=self._env.CLIENT
        )
        self._drone_marker = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self._drone_visual,
            basePosition=[0.0, 0.0, 1.0],
            physicsClientId=self._env.CLIENT
        )

        # Ground plane grid for reference
        self._create_ground_grid()

    def _create_ground_grid(self):
        """Create ground grid lines for spatial reference."""
        for i in range(-5, 6, 1):
            # X-axis lines
            p.addUserDebugLine(
                [i, -5, 0.01], [i, 5, 0.01],
                [0.5, 0.5, 0.5], 1,
                physicsClientId=self._env.CLIENT
            )
            # Y-axis lines
            p.addUserDebugLine(
                [-5, i, 0.01], [5, i, 0.01],
                [0.5, 0.5, 0.5], 1,
                physicsClientId=self._env.CLIENT
            )

    def _update_person_position(self):
        """Update person position with random walk."""
        if self._episode_step % 60 == 0:
            angle = self._rng.uniform(0, 2 * np.pi)
            self._person_vel = np.array([
                self._person_speed * np.cos(angle),
                self._person_speed * np.sin(angle),
                0.0
            ])

        dt = 1.0 / 30.0
        self._person_pos += self._person_vel * dt
        self._person_pos = np.clip(self._person_pos, -5.0, 5.0)
        self._person_pos[2] = 0.0

        # Update visual markers
        p.resetBasePositionAndOrientation(
            self._person_body,
            [self._person_pos[0], self._person_pos[1], 1.0],
            [0, 0, 0, 1],
            physicsClientId=self._env.CLIENT
        )
        p.resetBasePositionAndOrientation(
            self._head_body,
            [self._person_pos[0], self._person_pos[1], 2.2],
            [0, 0, 0, 1],
            physicsClientId=self._env.CLIENT
        )
        p.resetBasePositionAndOrientation(
            self._tag_body,
            [self._person_pos[0], self._person_pos[1], self._tag_height],
            [0, 0, 0, 1],
            physicsClientId=self._env.CLIENT
        )

        # Update target marker position
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

    def _get_target_position(self):
        """Get ideal drone position (behind person at target distance/height)."""
        tag_pos = self._get_tag_position()
        # Target position is behind the person
        target_pos = tag_pos.copy()
        target_pos[0] -= self._target_distance
        target_pos[2] = self._target_height
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
            distance = azimuth = elevation = signal_strength = 0.0

        return np.array([distance, azimuth, elevation, signal_strength, float(detected)], dtype=np.float32)

    def _project_to_screen(self, point_3d, view_matrix, proj_matrix, width, height):
        """Project 3D point to 2D screen coordinates."""
        # Convert to homogeneous coordinates
        point = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])

        # Apply view matrix
        view = np.array(view_matrix).reshape(4, 4).T
        point_view = view @ point

        # Apply projection matrix
        proj = np.array(proj_matrix).reshape(4, 4).T
        point_proj = proj @ point_view

        # Perspective division
        if point_proj[3] != 0:
            point_ndc = point_proj[:3] / point_proj[3]
        else:
            return None

        # Check if point is behind camera
        if point_proj[3] < 0:
            return None

        # Convert to screen coordinates
        x = int((point_ndc[0] + 1) * width / 2)
        y = int((1 - point_ndc[1]) * height / 2)

        # Check if on screen
        if 0 <= x < width and 0 <= y < height:
            return (x, y)
        return None

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

        # Project 3D positions to 2D and draw markers
        # Person position (magenta circle)
        person_screen = self._project_to_screen(
            [self._person_pos[0], self._person_pos[1], 1.0],
            view_matrix, proj_matrix, width, height
        )
        if person_screen:
            cv2.circle(frame, person_screen, 40, (255, 0, 255), 4)  # Magenta
            cv2.putText(frame, "PERSON", (person_screen[0]-30, person_screen[1]-50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Tag position (red cross)
        tag_screen = self._project_to_screen(tag_pos_3d, view_matrix, proj_matrix, width, height)
        if tag_screen:
            size = 30
            cv2.line(frame, (tag_screen[0]-size, tag_screen[1]), (tag_screen[0]+size, tag_screen[1]), (0, 0, 255), 4)
            cv2.line(frame, (tag_screen[0], tag_screen[1]-size), (tag_screen[0], tag_screen[1]+size), (0, 0, 255), 4)
            cv2.putText(frame, "TAG", (tag_screen[0]+35, tag_screen[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Target position (where drone should be - cyan circle)
        target_pos = self._get_target_position()
        target_screen = self._project_to_screen(target_pos, view_matrix, proj_matrix, width, height)
        if target_screen:
            cv2.circle(frame, target_screen, 25, (255, 255, 0), 3)  # Cyan
            cv2.putText(frame, "TARGET", (target_screen[0]-25, target_screen[1]-35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Draw line from drone to tag (yellow - tracking line)
        drone_screen = self._project_to_screen(drone_pos, view_matrix, proj_matrix, width, height)
        if tag_screen and drone_screen:
            cv2.line(frame, drone_screen, tag_screen, (0, 255, 255), 2)

        # Draw distance bar (right side of screen)
        self._draw_distance_bar(frame, drone_pos, tag_pos_3d, width, height)

        # Draw radar FOV indicator (top right)
        self._draw_radar_indicator(frame, drone_pos, tag_pos_3d, width)

        return frame

    def _draw_distance_bar(self, frame, drone_pos, tag_pos, width, height):
        """Draw a graphical distance bar."""
        import cv2
        actual_dist = np.linalg.norm(np.array(drone_pos) - np.array(tag_pos))
        target_dist = self._target_distance

        # Bar position (right side)
        bar_x = width - 60
        bar_top = 150
        bar_height = 300
        bar_width = 30

        # Background
        cv2.rectangle(frame, (bar_x - 5, bar_top - 30), (bar_x + bar_width + 5, bar_top + bar_height + 40),
                     (0, 0, 0), -1)
        cv2.putText(frame, "DIST", (bar_x - 5, bar_top - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw bar outline
        cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_width, bar_top + bar_height),
                     (100, 100, 100), 2)

        # Target zone (green band at target distance)
        # Map distance 0-10m to bar height
        max_dist = 10.0
        target_y = bar_top + int((target_dist / max_dist) * bar_height)
        zone_height = 30
        cv2.rectangle(frame, (bar_x, target_y - zone_height//2),
                     (bar_x + bar_width, target_y + zone_height//2),
                     (0, 150, 0), -1)

        # Current distance marker
        current_y = bar_top + int(min(actual_dist, max_dist) / max_dist * bar_height)
        error = abs(actual_dist - target_dist)
        if error < 0.5:
            color = (0, 255, 0)  # Green - good
        elif error < 1.5:
            color = (0, 165, 255)  # Orange - ok
        else:
            color = (0, 0, 255)  # Red - far

        cv2.rectangle(frame, (bar_x - 8, current_y - 5), (bar_x + bar_width + 8, current_y + 5),
                     color, -1)

        # Distance text
        cv2.putText(frame, f"{actual_dist:.1f}m", (bar_x - 10, bar_top + bar_height + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"({target_dist:.1f})", (bar_x - 10, bar_top + bar_height + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)

    def _draw_radar_indicator(self, frame, drone_pos, tag_pos, width):
        """Draw radar detection indicator."""
        import cv2
        # Radar indicator box (top right)
        box_x = width - 180
        box_y = 10
        box_w = 170
        box_h = 100

        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (100, 100, 100), 1)
        cv2.putText(frame, "RADAR", (box_x + 55, box_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Calculate radar parameters
        rel_pos = np.array(tag_pos) - np.array(drone_pos)
        distance = np.linalg.norm(rel_pos)
        azimuth = np.degrees(np.arctan2(rel_pos[1], rel_pos[0]))
        elevation = np.degrees(np.arctan2(rel_pos[2], np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)))

        # Check if in FOV
        in_range = self._radar_min_range < distance < self._radar_max_range
        in_azimuth = abs(azimuth) < self._radar_fov_azimuth / 2
        in_elevation = abs(elevation) < self._radar_fov_elevation / 2
        detected = in_range and in_azimuth and in_elevation

        # Status
        status_color = (0, 255, 0) if detected else (0, 0, 255)
        status_text = "TRACKING" if detected else "LOST"
        cv2.putText(frame, status_text, (box_x + 45, box_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Mini FOV diagram
        center_x = box_x + box_w // 2
        center_y = box_y + 75
        fov_radius = 30

        # Draw FOV arc
        cv2.ellipse(frame, (center_x, center_y), (fov_radius, fov_radius),
                   0, -self._radar_fov_azimuth/2, self._radar_fov_azimuth/2, (100, 100, 100), 1)

        # Draw target direction
        if distance > 0.1:
            angle_rad = np.radians(azimuth)
            target_x = int(center_x + np.cos(angle_rad) * min(fov_radius, fov_radius * distance / 5))
            target_y = int(center_y - np.sin(angle_rad) * min(fov_radius, fov_radius * distance / 5))
            dot_color = (0, 255, 0) if detected else (0, 0, 255)
            cv2.circle(frame, (target_x, center_y), 5, dot_color, -1)

    @property
    def observation_space(self):
        import gym
        # v2 state: drone(12) + extra(60) + radar(5) + velocity(3) + target(3) + curriculum(1) = 84
        return gym.spaces.Dict({
            "state": gym.spaces.Box(-np.inf, np.inf, (84,), dtype=np.float32)
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

        # v2 extended obs: obs_flat(72) + radar(5) + velocity(3) + target(3) + curriculum(1) = 84
        extended_obs = np.concatenate([
            obs_flat,
            radar_obs,
            np.zeros(3, dtype=np.float32),  # velocity estimate
            np.array([self._target_distance, self._target_height, 0.0], dtype=np.float32),  # target info
            np.array([float(self._current_stage)], dtype=np.float32),  # curriculum stage
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
            self._update_person_position()
            tag_pos = self._get_tag_position()
            radar_obs = self._simulate_radar_detection(drone_pos, tag_pos)

            # v2 reward calculation
            detected = radar_obs[4] > 0.5

            # Survival bonus
            reward += 0.5

            if not detected:
                reward -= 0.5  # Reduced penalty (v2)
            else:
                # Distance reward (shaping)
                dist_err = np.abs(radar_obs[0] - self._target_distance)
                reward += max(0, 2.0 - dist_err)

                # Height reward
                height_err = np.abs(drone_pos[2] - self._target_height)
                reward += max(0, 1.0 - height_err)

            if terminated or truncated:
                break

        distance_error = 0.0
        if radar_obs[4] > 0.5:
            distance_error = radar_obs[0] - self._target_distance

        extended_obs = np.concatenate([
            obs_flat,
            radar_obs,
            self._person_vel.astype(np.float32),
            np.array([self._target_distance, self._target_height, distance_error], dtype=np.float32),
            np.array([float(self._current_stage)], dtype=np.float32),
        ])

        info = {
            "drone_pos": drone_pos,
            "tag_pos": tag_pos,
            "distance": np.linalg.norm(drone_pos - tag_pos),
            "detected": detected,
            "target_pos": self._get_target_position(),
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
            (f"MilliSign Drone Follow - Episode {ep}", color, 1.0),
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

        # Legend at bottom
        legend_y = frame.shape[0] - 30
        cv2.putText(frame, "Yellow=Drone  Magenta=Person  Red=Tag  Cyan=Target",
                    (10, legend_y), font, 0.6, (200, 200, 200), 1)

    except ImportError:
        pass
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="/workspace/logdir/millisignv2")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output", type=str, default="millisignv2_eval.mp4")
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
    # Check if using improved model
    if "improved" in str(args.logdir):
        recursive_update(config, configs["millisignv2_improved"])
    else:
        recursive_update(config, configs["millisignv2"])
    config = to_attrdict(config)
    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.num_actions = 4

    print("Creating environment...")
    env = MilliSignEnvV2Video(seed=42, size=(1280, 720))

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
            frame = add_overlay(frame, ep + 1, step, episode_reward, info["distance"], info["detected"],
                               target_dist=2.5, drone_pos=info["drone_pos"], tag_pos=info["tag_pos"])
            frames.append(frame)

            if step % 30 == 0:
                print(f"  Step {step}: Reward={episode_reward:.1f}, Dist={info['distance']:.2f}m, Detected={info['detected']}")

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
