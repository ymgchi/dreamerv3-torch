"""
DreamerV3 wrapper for MilliSign-based person following task.

Based on the MilliSign paper (ACM MobiCom 2023):
"mmWave-Based Passive Signs for Guiding UAVs in Poor Visibility Conditions"

This environment simulates a drone following a person wearing a mmWave
reflector tag. The tag provides reliable 3D position information even
in poor visibility conditions (fog, rain, night).
"""
import gym
import numpy as np


class PyBulletDroneMilliSign:
    """DreamerV3 compatible wrapper for MilliSign-based person following."""

    metadata = {}

    def __init__(self, task="follow", action_repeat=1, size=(64, 64), seed=0):
        """
        Initialize the MilliSign person following environment.

        Parameters
        ----------
        task : str
            Task type: "follow" (follow person), "approach" (approach and hover)
        action_repeat : int
            Number of times to repeat each action
        size : tuple
            Image observation size (width, height)
        seed : int
            Random seed
        """
        from gym_pybullet_drones.envs.HoverAviary import HoverAviary
        from gym_pybullet_drones.utils.enums import ActionType, ObservationType

        self._action_repeat = action_repeat
        self._size = size
        self._task = task
        self._rng = np.random.RandomState(seed)

        # Create the base environment
        self._env = HoverAviary(
            gui=False,
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            pyb_freq=240,
            ctrl_freq=30,
        )

        self.reward_range = [-np.inf, np.inf]
        self._step_count = 0
        self._episode_step = 0

        # MilliSign radar parameters (based on paper)
        self._radar_max_range = 15.0  # meters (paper shows 10m+ reliable)
        self._radar_min_range = 0.5   # meters
        self._radar_fov_azimuth = 60  # degrees (±30°)
        self._radar_fov_elevation = 60  # degrees (±30°)
        self._radar_noise_distance = 0.01  # 10mm noise (paper: 3.5mm accuracy)
        self._radar_noise_angle = 1.0  # 1 degree noise

        # Person/target parameters
        self._person_pos = np.array([0.0, 0.0, 0.0])  # Ground level
        self._person_vel = np.array([0.0, 0.0, 0.0])
        self._person_speed = 1.0  # m/s walking speed
        self._tag_height = 1.5  # Tag worn at chest height

        # Following parameters
        self._target_distance = 3.0  # Desired following distance
        self._target_height = 2.0    # Desired flight height above ground
        self._distance_threshold = 0.5  # Acceptable distance error

        # Movement patterns for person
        self._movement_patterns = {
            "stationary": self._person_stationary,
            "walk_line": self._person_walk_line,
            "walk_circle": self._person_walk_circle,
            "walk_random": self._person_walk_random,
            "walk_zigzag": self._person_walk_zigzag,
        }
        self._current_pattern = "walk_random"

    def _person_stationary(self, t):
        """Person stands still."""
        return np.array([2.0, 0.0, 0.0])

    def _person_walk_line(self, t):
        """Person walks in a straight line."""
        speed = 0.5  # m/s
        return np.array([2.0 + speed * t, 0.0, 0.0])

    def _person_walk_circle(self, t):
        """Person walks in a circle."""
        radius = 3.0
        omega = 0.3  # rad/s
        angle = omega * t
        return np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            0.0
        ])

    def _person_walk_random(self, t):
        """Person walks with random direction changes."""
        # Update velocity occasionally
        if self._episode_step % 60 == 0:  # Change direction every 2 seconds
            angle = self._rng.uniform(0, 2 * np.pi)
            speed = self._rng.uniform(0.3, 0.8)
            self._person_vel = np.array([
                speed * np.cos(angle),
                speed * np.sin(angle),
                0.0
            ])

        # Update position
        dt = 1.0 / 30.0  # Control frequency
        self._person_pos += self._person_vel * dt

        # Keep person in bounds
        self._person_pos = np.clip(self._person_pos, -5.0, 5.0)
        self._person_pos[2] = 0.0  # Keep on ground

        return self._person_pos.copy()

    def _person_walk_zigzag(self, t):
        """Person walks in zigzag pattern."""
        speed = 0.5
        x = speed * t
        y = 2.0 * np.sin(0.5 * t)
        return np.array([x, y, 0.0])

    def _get_tag_position(self):
        """Get the position of the MilliSign tag (at person's chest)."""
        person_pos = self._update_person_position()
        tag_pos = person_pos.copy()
        tag_pos[2] = self._tag_height  # Tag at chest height
        return tag_pos

    def _update_person_position(self):
        """Update and return person position based on movement pattern."""
        t = self._episode_step / 30.0  # Time in seconds
        pattern_func = self._movement_patterns.get(
            self._current_pattern, self._person_walk_random
        )
        return pattern_func(t)

    def _simulate_radar_detection(self, drone_pos, tag_pos):
        """
        Simulate MilliSign radar detection with Root-MUSIC processing.

        Returns radar observation: [distance, azimuth, elevation, signal_strength, detected]
        """
        # Calculate relative position
        rel_pos = tag_pos - drone_pos

        # Convert to spherical coordinates
        distance = np.linalg.norm(rel_pos)
        azimuth = np.arctan2(rel_pos[1], rel_pos[0])  # Horizontal angle
        elevation = np.arctan2(rel_pos[2], np.sqrt(rel_pos[0]**2 + rel_pos[1]**2))

        # Check if tag is within radar FOV and range
        detected = True
        if distance > self._radar_max_range or distance < self._radar_min_range:
            detected = False
        if np.abs(np.degrees(azimuth)) > self._radar_fov_azimuth / 2:
            detected = False
        if np.abs(np.degrees(elevation)) > self._radar_fov_elevation / 2:
            detected = False

        # Add realistic noise (based on paper's accuracy)
        if detected:
            distance += self._rng.normal(0, self._radar_noise_distance)
            azimuth += np.radians(self._rng.normal(0, self._radar_noise_angle))
            elevation += np.radians(self._rng.normal(0, self._radar_noise_angle))

            # Signal strength based on distance (radar equation: 1/r^4)
            signal_strength = np.clip(1.0 / (distance / 5.0) ** 2, 0, 1)
        else:
            # No detection - return zeros
            distance = 0.0
            azimuth = 0.0
            elevation = 0.0
            signal_strength = 0.0

        return np.array([
            distance,
            azimuth,
            elevation,
            signal_strength,
            float(detected)
        ], dtype=np.float32)

    def _calculate_reward(self, drone_pos, tag_pos, radar_obs):
        """Calculate reward for person following task."""
        reward = 0.0

        # Get detection status
        detected = radar_obs[4] > 0.5

        if not detected:
            # Penalty for losing track of person
            reward -= 2.0
            return reward

        distance = radar_obs[0]

        # Distance reward: maintain target following distance
        distance_error = np.abs(distance - self._target_distance)
        if distance_error < self._distance_threshold:
            reward += 2.0  # Bonus for good following distance
        else:
            reward += max(0, 2.0 - distance_error)

        # Height reward: maintain appropriate altitude
        height_error = np.abs(drone_pos[2] - self._target_height)
        reward += max(0, 1.0 - height_error)

        # Centering reward: keep tag centered in FOV
        azimuth_error = np.abs(radar_obs[1])  # radians
        elevation_error = np.abs(radar_obs[2])
        centering_reward = max(0, 1.0 - (azimuth_error + elevation_error))
        reward += centering_reward

        # Smoothness penalty (from previous action) - handled elsewhere

        return reward

    @property
    def observation_space(self):
        """Return observation space compatible with DreamerV3."""
        spaces = {}

        # Drone state (12 dims from HoverAviary)
        orig_space = self._env.observation_space
        if hasattr(orig_space, 'shape'):
            flat_size = int(np.prod(orig_space.shape))
        else:
            flat_size = 12

        # Extended state:
        # - drone_state (12): position, orientation, velocities
        # - radar_detection (5): distance, azimuth, elevation, signal, detected
        # - relative_velocity (3): estimated person velocity
        # - target_info (3): target distance, height, following error
        state_size = flat_size + 5 + 3 + 3

        spaces["state"] = gym.spaces.Box(
            -np.inf, np.inf, (state_size,), dtype=np.float32
        )

        # Optional image observation
        spaces["image"] = gym.spaces.Box(
            0, 255, self._size + (3,), dtype=np.uint8
        )

        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        """Return action space."""
        orig_space = self._env.action_space
        return gym.spaces.Box(
            orig_space.low.flatten(),
            orig_space.high.flatten(),
            dtype=np.float32
        )

    def step(self, action):
        """Execute action and return observation."""
        assert np.isfinite(action).all(), f"Invalid action: {action}"

        if len(action.shape) == 1:
            action = action.reshape(self._env.action_space.shape)

        reward = 0.0
        done = False
        info = {}

        for _ in range(self._action_repeat):
            obs, r, terminated, truncated, env_info = self._env.step(action)
            self._episode_step += 1

            # Get positions
            drone_pos = obs.flatten()[:3]
            tag_pos = self._get_tag_position()

            # Simulate radar detection
            radar_obs = self._simulate_radar_detection(drone_pos, tag_pos)

            # Calculate custom reward
            step_reward = self._calculate_reward(drone_pos, tag_pos, radar_obs)
            reward += step_reward

            done = terminated or truncated
            if done:
                break

        self._step_count += 1

        # Build observation
        obs_flat = np.array(obs, dtype=np.float32).flatten()
        drone_pos = obs_flat[:3]

        # Estimate person velocity (simple finite difference)
        person_vel_est = self._person_vel if hasattr(self, '_person_vel') else np.zeros(3)

        # Target info
        target_info = np.array([
            self._target_distance,
            self._target_height,
            radar_obs[0] - self._target_distance if radar_obs[4] > 0.5 else 0.0
        ], dtype=np.float32)

        # Combine all observations
        extended_obs = np.concatenate([
            obs_flat,           # Drone state (12)
            radar_obs,          # Radar detection (5)
            person_vel_est,     # Person velocity estimate (3)
            target_info,        # Target info (3)
        ])

        obs_dict = {
            "state": extended_obs.astype(np.float32),
            "image": self.render(),
            "is_terminal": terminated,
            "is_first": False,
        }

        info["discount"] = np.array(0.0 if terminated else 1.0, np.float32)
        info["tag_pos"] = tag_pos.copy()
        info["radar_detection"] = radar_obs.copy()
        info["distance_to_tag"] = np.linalg.norm(drone_pos - tag_pos)

        return obs_dict, reward, done, info

    def reset(self):
        """Reset environment."""
        obs, info = self._env.reset()
        self._step_count = 0
        self._episode_step = 0

        # Reset person position and velocity
        self._person_pos = np.array([
            self._rng.uniform(2.0, 4.0),
            self._rng.uniform(-1.0, 1.0),
            0.0
        ])
        self._person_vel = np.zeros(3)

        # Randomly select movement pattern
        patterns = list(self._movement_patterns.keys())
        self._current_pattern = self._rng.choice(patterns)

        # Get initial observations
        obs_flat = np.array(obs, dtype=np.float32).flatten()
        drone_pos = obs_flat[:3]
        tag_pos = self._get_tag_position()
        radar_obs = self._simulate_radar_detection(drone_pos, tag_pos)

        person_vel_est = np.zeros(3, dtype=np.float32)
        target_info = np.array([
            self._target_distance,
            self._target_height,
            0.0
        ], dtype=np.float32)

        extended_obs = np.concatenate([
            obs_flat,
            radar_obs,
            person_vel_est,
            target_info,
        ])

        obs_dict = {
            "state": extended_obs.astype(np.float32),
            "image": self.render(),
            "is_terminal": False,
            "is_first": True,
        }

        return obs_dict

    def render(self, *args, **kwargs):
        """Render environment to RGB array with tag visualization."""
        try:
            import pybullet as p

            width, height = self._size

            # Camera follows drone, looks toward tag
            drone_pos = self._env.pos[0] if hasattr(self._env, 'pos') else [0, 0, 1]
            tag_pos = self._get_tag_position()

            # Camera position behind and above drone
            cam_pos = [
                drone_pos[0] - 2,
                drone_pos[1] - 2,
                drone_pos[2] + 1
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

        except Exception:
            return np.zeros(self._size + (3,), dtype=np.uint8)

    def close(self):
        """Close the environment."""
        self._env.close()
