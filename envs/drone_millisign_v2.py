"""
DreamerV3 wrapper for MilliSign-based person following task (v2).

Improvements over v1:
- Velocity control instead of RPM (easier to learn)
- Curriculum learning (stationary -> slow -> random movement)
- Better initial conditions (start near target position)
- Denser rewards with survival bonus
- Gradual difficulty increase based on training progress
"""
import gym
import numpy as np


class PyBulletDroneMilliSignV2:
    """DreamerV3 compatible wrapper for MilliSign-based person following (v2)."""

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

        # Use velocity control instead of RPM (much easier to learn)
        self._env = HoverAviary(
            gui=False,
            obs=ObservationType.KIN,
            act=ActionType.VEL,  # Changed from RPM to VEL
            pyb_freq=240,
            ctrl_freq=30,
        )

        self.reward_range = [-np.inf, np.inf]
        self._step_count = 0
        self._episode_step = 0
        self._total_steps = 0  # For curriculum learning

        # MilliSign radar parameters (based on AWR1843 specs)
        self._radar_max_range = 15.0
        self._radar_min_range = 0.3
        self._radar_fov_azimuth = 120   # AWR1843: ±60° (120° total)
        self._radar_fov_elevation = 40  # AWR1843: ±20° (40° total)
        self._radar_noise_distance = 0.01  # 10mm noise
        self._radar_noise_angle = 1.0  # 1 degree noise (Root-MUSIC accuracy)

        # Person/target parameters
        self._person_pos = np.array([0.0, 0.0, 0.0])
        self._person_vel = np.array([0.0, 0.0, 0.0])
        self._person_speed = 1.0
        self._tag_height = 1.5

        # Following parameters (relaxed)
        self._target_distance = 2.5  # Reduced from 3.0
        self._target_height = 1.8    # Reduced from 2.0
        self._distance_threshold = 1.0  # Increased from 0.5 (more forgiving)

        # Curriculum learning parameters
        self._curriculum_stages = [
            {"name": "stationary", "steps": 100000, "person_speed": 0.0},
            {"name": "slow", "steps": 200000, "person_speed": 0.3},
            {"name": "medium", "steps": 350000, "person_speed": 0.5},
            {"name": "full", "steps": np.inf, "person_speed": 0.8},
        ]
        self._current_stage = 0

        # Movement patterns for person
        self._movement_patterns = {
            "stationary": self._person_stationary,
            "walk_line": self._person_walk_line,
            "walk_circle": self._person_walk_circle,
            "walk_random": self._person_walk_random,
            "walk_zigzag": self._person_walk_zigzag,
        }
        self._current_pattern = "stationary"

        # Previous position for velocity estimation
        self._prev_person_pos = None

    def _get_curriculum_stage(self):
        """Get current curriculum stage based on total steps."""
        for i, stage in enumerate(self._curriculum_stages):
            if self._total_steps < stage["steps"]:
                return i
        return len(self._curriculum_stages) - 1

    def _get_person_speed_multiplier(self):
        """Get person speed multiplier based on curriculum stage."""
        stage_idx = self._get_curriculum_stage()
        return self._curriculum_stages[stage_idx]["person_speed"]

    def _person_stationary(self, t):
        """Person stands still."""
        return self._person_pos.copy()

    def _person_walk_line(self, t):
        """Person walks in a straight line."""
        speed = 0.5 * self._get_person_speed_multiplier()
        dt = 1.0 / 30.0
        self._person_pos[0] += speed * dt
        self._person_pos = np.clip(self._person_pos, -5.0, 5.0)
        self._person_pos[2] = 0.0
        return self._person_pos.copy()

    def _person_walk_circle(self, t):
        """Person walks in a circle."""
        radius = 2.5
        omega = 0.2 * self._get_person_speed_multiplier()
        angle = omega * t
        self._person_pos = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            0.0
        ])
        return self._person_pos.copy()

    def _person_walk_random(self, t):
        """Person walks with random direction changes."""
        speed_mult = self._get_person_speed_multiplier()

        # Update velocity occasionally
        if self._episode_step % 90 == 0:  # Change direction every 3 seconds
            angle = self._rng.uniform(0, 2 * np.pi)
            speed = self._rng.uniform(0.2, 0.6) * speed_mult
            self._person_vel = np.array([
                speed * np.cos(angle),
                speed * np.sin(angle),
                0.0
            ])

        # Update position
        dt = 1.0 / 30.0
        self._person_pos += self._person_vel * dt

        # Keep person in bounds
        self._person_pos = np.clip(self._person_pos, -4.0, 4.0)
        self._person_pos[2] = 0.0

        return self._person_pos.copy()

    def _person_walk_zigzag(self, t):
        """Person walks in zigzag pattern."""
        speed = 0.4 * self._get_person_speed_multiplier()
        dt = 1.0 / 30.0
        self._person_pos[0] += speed * dt
        self._person_pos[1] = 1.5 * np.sin(0.3 * t)
        self._person_pos = np.clip(self._person_pos, -4.0, 4.0)
        self._person_pos[2] = 0.0
        return self._person_pos.copy()

    def _get_tag_position(self):
        """Get the position of the MilliSign tag (at person's chest)."""
        person_pos = self._update_person_position()
        tag_pos = person_pos.copy()
        tag_pos[2] = self._tag_height
        return tag_pos

    def _update_person_position(self):
        """Update and return person position based on movement pattern."""
        t = self._episode_step / 30.0
        pattern_func = self._movement_patterns.get(
            self._current_pattern, self._person_stationary
        )
        return pattern_func(t)

    def _simulate_radar_detection(self, drone_pos, tag_pos):
        """
        Simulate MilliSign radar detection with Root-MUSIC processing.
        """
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
            # Partial observation even when not fully detected (degraded)
            distance = np.clip(distance, 0, self._radar_max_range)
            signal_strength = 0.1  # Weak signal

        return np.array([
            distance,
            azimuth,
            elevation,
            signal_strength,
            float(detected)
        ], dtype=np.float32)

    def _calculate_reward(self, drone_pos, tag_pos, radar_obs, action):
        """Calculate reward for person following task (improved version)."""
        reward = 0.0

        # 1. Survival bonus (encourage staying alive)
        reward += 0.5

        detected = radar_obs[4] > 0.5
        distance = radar_obs[0]
        actual_distance = np.linalg.norm(drone_pos - tag_pos)

        # 2. Detection bonus/penalty (reduced penalty)
        if detected:
            reward += 0.5
        else:
            reward -= 0.5  # Reduced from -2.0

        # 3. Distance reward (shaped reward)
        distance_error = np.abs(actual_distance - self._target_distance)
        if distance_error < self._distance_threshold:
            # Good following distance
            reward += 2.0 * (1.0 - distance_error / self._distance_threshold)
        else:
            # Gradual penalty based on distance
            reward += max(-1.0, 1.0 - distance_error / 5.0)

        # 4. Height reward (encourage correct altitude)
        height_error = np.abs(drone_pos[2] - self._target_height)
        if height_error < 0.5:
            reward += 1.0 * (1.0 - height_error / 0.5)
        else:
            reward -= min(1.0, height_error / 2.0)

        # 5. Centering reward (keep tag in view)
        if detected:
            azimuth_error = np.abs(radar_obs[1])
            elevation_error = np.abs(radar_obs[2])
            centering_score = max(0, 1.0 - (azimuth_error + elevation_error) / np.pi)
            reward += 0.5 * centering_score

        # 6. Smoothness reward (penalize jerky movements)
        action_magnitude = np.linalg.norm(action)
        if action_magnitude > 1.0:
            reward -= 0.1 * (action_magnitude - 1.0)

        # 7. Crash prevention (penalize extreme positions)
        if drone_pos[2] < 0.3:  # Too low
            reward -= 2.0
        if drone_pos[2] > 4.0:  # Too high
            reward -= 1.0
        if np.any(np.abs(drone_pos[:2]) > 6.0):  # Out of bounds
            reward -= 2.0

        return reward

    @property
    def observation_space(self):
        """Return observation space compatible with DreamerV3."""
        spaces = {}

        orig_space = self._env.observation_space
        if hasattr(orig_space, 'shape'):
            flat_size = int(np.prod(orig_space.shape))
        else:
            flat_size = 12

        # Extended state: drone(12) + radar(5) + velocity(3) + target(3) + curriculum(1)
        state_size = flat_size + 5 + 3 + 3 + 1

        spaces["state"] = gym.spaces.Box(
            -np.inf, np.inf, (state_size,), dtype=np.float32
        )

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
            self._total_steps += 1

            drone_pos = obs.flatten()[:3]
            tag_pos = self._get_tag_position()
            radar_obs = self._simulate_radar_detection(drone_pos, tag_pos)

            step_reward = self._calculate_reward(drone_pos, tag_pos, radar_obs, action.flatten())
            reward += step_reward

            done = terminated or truncated

            # Additional termination conditions
            if drone_pos[2] < 0.1:  # Crashed
                done = True
                reward -= 5.0
            if np.any(np.abs(drone_pos[:2]) > 7.0):  # Way out of bounds
                done = True
                reward -= 5.0

            if done:
                break

        self._step_count += 1

        # Build observation
        obs_flat = np.array(obs, dtype=np.float32).flatten()
        drone_pos = obs_flat[:3]

        # Estimate person velocity
        if self._prev_person_pos is not None:
            person_vel_est = (self._person_pos - self._prev_person_pos) * 30.0
        else:
            person_vel_est = np.zeros(3)
        self._prev_person_pos = self._person_pos.copy()

        # Target info
        target_info = np.array([
            self._target_distance,
            self._target_height,
            radar_obs[0] - self._target_distance if radar_obs[4] > 0.5 else 0.0
        ], dtype=np.float32)

        # Curriculum stage info
        curriculum_info = np.array([self._get_curriculum_stage() / 3.0], dtype=np.float32)

        extended_obs = np.concatenate([
            obs_flat,
            radar_obs,
            person_vel_est.astype(np.float32),
            target_info,
            curriculum_info,
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
        info["curriculum_stage"] = self._get_curriculum_stage()

        return obs_dict, reward, done, info

    def reset(self):
        """Reset environment with curriculum-aware initialization."""
        obs, info = self._env.reset()
        self._step_count = 0
        self._episode_step = 0
        self._prev_person_pos = None

        # Update curriculum stage
        stage_idx = self._get_curriculum_stage()
        stage = self._curriculum_stages[stage_idx]

        # Initialize person position (closer in early stages)
        if stage_idx == 0:
            # Stationary stage: fixed position
            self._person_pos = np.array([2.0, 0.0, 0.0])
            self._current_pattern = "stationary"
        elif stage_idx == 1:
            # Slow stage: simple patterns
            self._person_pos = np.array([
                self._rng.uniform(1.5, 2.5),
                self._rng.uniform(-0.5, 0.5),
                0.0
            ])
            self._current_pattern = self._rng.choice(["stationary", "walk_line", "walk_circle"])
        else:
            # Later stages: more variety
            self._person_pos = np.array([
                self._rng.uniform(1.5, 3.0),
                self._rng.uniform(-1.0, 1.0),
                0.0
            ])
            patterns = list(self._movement_patterns.keys())
            self._current_pattern = self._rng.choice(patterns)

        self._person_vel = np.zeros(3)

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
        curriculum_info = np.array([stage_idx / 3.0], dtype=np.float32)

        extended_obs = np.concatenate([
            obs_flat,
            radar_obs,
            person_vel_est,
            target_info,
            curriculum_info,
        ])

        obs_dict = {
            "state": extended_obs.astype(np.float32),
            "image": self.render(),
            "is_terminal": False,
            "is_first": True,
        }

        return obs_dict

    def render(self, *args, **kwargs):
        """Render environment to RGB array."""
        try:
            import pybullet as p

            width, height = self._size

            drone_pos = self._env.pos[0] if hasattr(self._env, 'pos') else [0, 0, 1]
            tag_pos = self._get_tag_position()

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

    def set_total_steps(self, steps):
        """Set total training steps (for curriculum learning)."""
        self._total_steps = steps
