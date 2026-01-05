"""
DreamerV3 wrapper for MilliSign-based person following task (v3).

Improvements over v2:
- Enhanced reward function with detection streak bonus
- Better curriculum learning with more high-speed focus
- Velocity prediction reward for fast tracking
- Re-detection bonus after losing track
"""
import gym
import numpy as np


class PyBulletDroneMilliSignV3:
    """DreamerV3 compatible wrapper for MilliSign-based person following (v3)."""

    metadata = {}

    def __init__(self, task="follow", action_repeat=1, size=(64, 64), seed=0):
        from gym_pybullet_drones.envs.HoverAviary import HoverAviary
        from gym_pybullet_drones.utils.enums import ActionType, ObservationType

        self._action_repeat = action_repeat
        self._size = size
        self._task = task
        self._rng = np.random.RandomState(seed)

        self._env = HoverAviary(
            gui=False,
            obs=ObservationType.KIN,
            act=ActionType.VEL,
            pyb_freq=240,
            ctrl_freq=30,
        )

        self.reward_range = [-np.inf, np.inf]
        self._step_count = 0
        self._episode_step = 0
        self._total_steps = 0

        # Radar parameters
        self._radar_max_range = 15.0
        self._radar_min_range = 0.3
        self._radar_fov_azimuth = 120
        self._radar_fov_elevation = 40
        self._radar_noise_distance = 0.01
        self._radar_noise_angle = 1.0

        # Person/target parameters
        self._person_pos = np.array([0.0, 0.0, 0.0])
        self._person_vel = np.array([0.0, 0.0, 0.0])
        self._tag_height = 1.5
        self._target_distance = 2.5
        self._target_height = 1.8
        self._distance_threshold = 1.0

        # Detection tracking for streak bonus
        self._detection_streak = 0
        self._last_detected = False
        self._lost_steps = 0

        # Previous positions for velocity
        self._prev_person_pos = None
        self._prev_drone_pos = None

        # Improved curriculum learning - more focus on high speed
        self._curriculum_stages = [
            {"name": "stationary", "steps": 50000, "person_speed": 0.0},
            {"name": "slow", "steps": 150000, "person_speed": 0.3},
            {"name": "medium", "steps": 300000, "person_speed": 0.5},
            {"name": "fast", "steps": 500000, "person_speed": 0.8},
            {"name": "very_fast", "steps": 750000, "person_speed": 1.0},
            {"name": "full", "steps": np.inf, "person_speed": 1.2},
        ]
        self._current_stage = 0

        self._movement_patterns = {
            "stationary": self._person_stationary,
            "walk_line": self._person_walk_line,
            "walk_circle": self._person_walk_circle,
            "walk_random": self._person_walk_random,
            "walk_zigzag": self._person_walk_zigzag,
            "walk_sudden": self._person_walk_sudden,  # New: sudden direction changes
        }
        self._current_pattern = "stationary"

    def _get_curriculum_stage(self):
        for i, stage in enumerate(self._curriculum_stages):
            if self._total_steps < stage["steps"]:
                return i
        return len(self._curriculum_stages) - 1

    def _get_person_speed_multiplier(self):
        stage_idx = self._get_curriculum_stage()
        return self._curriculum_stages[stage_idx]["person_speed"]

    def _person_stationary(self, t):
        return self._person_pos.copy()

    def _person_walk_line(self, t):
        speed = 0.5 * self._get_person_speed_multiplier()
        dt = 1.0 / 30.0
        self._person_pos[0] += speed * dt
        self._person_pos = np.clip(self._person_pos, -5.0, 5.0)
        self._person_pos[2] = 0.0
        return self._person_pos.copy()

    def _person_walk_circle(self, t):
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
        speed_mult = self._get_person_speed_multiplier()
        if self._episode_step % 60 == 0:  # More frequent changes
            angle = self._rng.uniform(0, 2 * np.pi)
            speed = self._rng.uniform(0.3, 0.8) * speed_mult
            self._person_vel = np.array([
                speed * np.cos(angle),
                speed * np.sin(angle),
                0.0
            ])
        dt = 1.0 / 30.0
        self._person_pos += self._person_vel * dt
        self._person_pos = np.clip(self._person_pos, -4.0, 4.0)
        self._person_pos[2] = 0.0
        return self._person_pos.copy()

    def _person_walk_zigzag(self, t):
        speed = 0.4 * self._get_person_speed_multiplier()
        dt = 1.0 / 30.0
        self._person_pos[0] += speed * dt
        self._person_pos[1] = 1.5 * np.sin(0.5 * t)
        self._person_pos = np.clip(self._person_pos, -4.0, 4.0)
        self._person_pos[2] = 0.0
        return self._person_pos.copy()

    def _person_walk_sudden(self, t):
        """Person with sudden direction changes - hard to track."""
        speed_mult = self._get_person_speed_multiplier()
        # More sudden changes
        if self._episode_step % 30 == 0:  # Every 1 second
            if self._rng.random() < 0.3:  # 30% chance of sudden change
                angle = self._rng.uniform(0, 2 * np.pi)
                speed = self._rng.uniform(0.5, 1.0) * speed_mult
                self._person_vel = np.array([
                    speed * np.cos(angle),
                    speed * np.sin(angle),
                    0.0
                ])
        dt = 1.0 / 30.0
        self._person_pos += self._person_vel * dt
        self._person_pos = np.clip(self._person_pos, -4.0, 4.0)
        self._person_pos[2] = 0.0
        return self._person_pos.copy()

    def _get_tag_position(self):
        person_pos = self._update_person_position()
        tag_pos = person_pos.copy()
        tag_pos[2] = self._tag_height
        return tag_pos

    def _update_person_position(self):
        t = self._episode_step / 30.0
        pattern_func = self._movement_patterns.get(
            self._current_pattern, self._person_stationary
        )
        return pattern_func(t)

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

        return np.array([
            distance,
            azimuth,
            elevation,
            signal_strength,
            float(detected)
        ], dtype=np.float32)

    def _calculate_reward(self, drone_pos, tag_pos, radar_obs, action):
        """Enhanced reward function with detection focus."""
        reward = 0.0
        detected = radar_obs[4] > 0.5
        actual_distance = np.linalg.norm(drone_pos - tag_pos)

        # 1. Survival bonus
        reward += 0.3

        # 2. Detection reward with streak bonus (KEY IMPROVEMENT)
        if detected:
            reward += 0.5
            self._detection_streak += 1
            # Streak bonus - reward continuous detection
            streak_bonus = min(self._detection_streak / 30.0, 1.0)  # Max at 1 second
            reward += 0.5 * streak_bonus

            # Re-detection bonus after losing track
            if not self._last_detected and self._lost_steps > 10:
                reward += 1.0  # Bonus for recovering
            self._lost_steps = 0
        else:
            # Penalty scales with how long we've been tracking
            penalty = min(0.5 + 0.1 * self._detection_streak / 30.0, 1.5)
            reward -= penalty
            self._detection_streak = 0
            self._lost_steps += 1

        self._last_detected = detected

        # 3. Distance reward
        distance_error = np.abs(actual_distance - self._target_distance)
        if distance_error < self._distance_threshold:
            reward += 2.0 * (1.0 - distance_error / self._distance_threshold)
        else:
            reward += max(-1.0, 1.0 - distance_error / 5.0)

        # 4. Height reward
        height_error = np.abs(drone_pos[2] - self._target_height)
        if height_error < 0.5:
            reward += 0.8 * (1.0 - height_error / 0.5)
        else:
            reward -= min(0.8, height_error / 2.0)

        # 5. Centering reward
        if detected:
            azimuth_error = np.abs(radar_obs[1])
            elevation_error = np.abs(radar_obs[2])
            centering_score = max(0, 1.0 - (azimuth_error + elevation_error) / np.pi)
            reward += 0.5 * centering_score

        # 6. Velocity matching reward (KEY IMPROVEMENT for fast tracking)
        if self._prev_drone_pos is not None and self._prev_person_pos is not None:
            drone_vel = (drone_pos - self._prev_drone_pos) * 30.0
            person_vel = (self._person_pos - self._prev_person_pos) * 30.0
            # Only XY velocity matters for following
            vel_match = 1.0 - min(1.0, np.linalg.norm(drone_vel[:2] - person_vel[:2]) / 2.0)
            reward += 0.3 * vel_match

        # 7. Anticipation reward - moving towards where person is going
        if detected and np.linalg.norm(self._person_vel[:2]) > 0.1:
            direction_to_target = tag_pos[:2] - drone_pos[:2]
            if np.linalg.norm(direction_to_target) > 0.1:
                direction_to_target = direction_to_target / np.linalg.norm(direction_to_target)
                person_dir = self._person_vel[:2] / np.linalg.norm(self._person_vel[:2])
                # Reward if drone is positioned to intercept
                alignment = np.dot(direction_to_target, person_dir)
                reward += 0.2 * max(0, alignment)

        # 8. Penalties
        action_magnitude = np.linalg.norm(action)
        if action_magnitude > 1.0:
            reward -= 0.1 * (action_magnitude - 1.0)

        if drone_pos[2] < 0.3:
            reward -= 2.0
        if drone_pos[2] > 4.0:
            reward -= 1.0
        if np.any(np.abs(drone_pos[:2]) > 6.0):
            reward -= 2.0

        return reward

    @property
    def observation_space(self):
        spaces = {}
        orig_space = self._env.observation_space
        if hasattr(orig_space, 'shape'):
            flat_size = int(np.prod(orig_space.shape))
        else:
            flat_size = 12

        # Extended state: drone(12) + radar(5) + velocity(3) + target(3) + curriculum(1) + streak(1)
        state_size = flat_size + 5 + 3 + 3 + 1 + 1

        spaces["state"] = gym.spaces.Box(
            -np.inf, np.inf, (state_size,), dtype=np.float32
        )
        spaces["image"] = gym.spaces.Box(
            0, 255, self._size + (3,), dtype=np.uint8
        )
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        orig_space = self._env.action_space
        return gym.spaces.Box(
            orig_space.low.flatten(),
            orig_space.high.flatten(),
            dtype=np.float32
        )

    def step(self, action):
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

            # Update previous positions
            self._prev_drone_pos = drone_pos.copy()
            self._prev_person_pos = self._person_pos.copy()

            done = terminated or truncated
            if drone_pos[2] < 0.1:
                done = True
                reward -= 5.0
            if np.any(np.abs(drone_pos[:2]) > 7.0):
                done = True
                reward -= 5.0

            if done:
                break

        self._step_count += 1

        obs_flat = np.array(obs, dtype=np.float32).flatten()
        drone_pos = obs_flat[:3]

        if self._prev_person_pos is not None:
            person_vel_est = (self._person_pos - self._prev_person_pos) * 30.0
        else:
            person_vel_est = np.zeros(3)

        target_info = np.array([
            self._target_distance,
            self._target_height,
            radar_obs[0] - self._target_distance if radar_obs[4] > 0.5 else 0.0
        ], dtype=np.float32)

        curriculum_info = np.array([self._get_curriculum_stage() / 5.0], dtype=np.float32)
        streak_info = np.array([min(self._detection_streak / 60.0, 1.0)], dtype=np.float32)

        extended_obs = np.concatenate([
            obs_flat,
            radar_obs,
            person_vel_est.astype(np.float32),
            target_info,
            curriculum_info,
            streak_info,
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
        info["detection_streak"] = self._detection_streak

        return obs_dict, reward, done, info

    def reset(self):
        obs, info = self._env.reset()
        self._step_count = 0
        self._episode_step = 0
        self._prev_person_pos = None
        self._prev_drone_pos = None
        self._detection_streak = 0
        self._last_detected = False
        self._lost_steps = 0

        stage_idx = self._get_curriculum_stage()
        stage = self._curriculum_stages[stage_idx]

        # More varied initial positions based on stage
        if stage_idx <= 1:
            self._person_pos = np.array([2.0, 0.0, 0.0])
            self._current_pattern = "stationary" if stage_idx == 0 else self._rng.choice(["stationary", "walk_line"])
        elif stage_idx <= 3:
            self._person_pos = np.array([
                self._rng.uniform(1.5, 2.5),
                self._rng.uniform(-0.5, 0.5),
                0.0
            ])
            self._current_pattern = self._rng.choice(["walk_line", "walk_circle", "walk_random"])
        else:
            # Later stages: harder patterns including sudden
            self._person_pos = np.array([
                self._rng.uniform(1.5, 3.0),
                self._rng.uniform(-1.0, 1.0),
                0.0
            ])
            self._current_pattern = self._rng.choice(["walk_random", "walk_zigzag", "walk_sudden"])

        self._person_vel = np.zeros(3)

        obs_flat = np.array(obs, dtype=np.float32).flatten()
        drone_pos = obs_flat[:3]
        tag_pos = self._get_tag_position()
        radar_obs = self._simulate_radar_detection(drone_pos, tag_pos)

        person_vel_est = np.zeros(3, dtype=np.float32)
        target_info = np.array([self._target_distance, self._target_height, 0.0], dtype=np.float32)
        curriculum_info = np.array([stage_idx / 5.0], dtype=np.float32)
        streak_info = np.array([0.0], dtype=np.float32)

        extended_obs = np.concatenate([
            obs_flat,
            radar_obs,
            person_vel_est,
            target_info,
            curriculum_info,
            streak_info,
        ])

        obs_dict = {
            "state": extended_obs.astype(np.float32),
            "image": self.render(),
            "is_terminal": False,
            "is_first": True,
        }

        return obs_dict

    def render(self, *args, **kwargs):
        try:
            import pybullet as p
            width, height = self._size
            drone_pos = self._env.pos[0] if hasattr(self._env, 'pos') else [0, 0, 1]
            tag_pos = self._get_tag_position()

            cam_pos = [drone_pos[0] - 2, drone_pos[1] - 2, drone_pos[2] + 1]
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=cam_pos,
                cameraTargetPosition=drone_pos,
                cameraUpVector=[0, 0, 1],
                physicsClientId=self._env.CLIENT
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=width / height, nearVal=0.1, farVal=100.0,
                physicsClientId=self._env.CLIENT
            )
            _, _, rgb, _, _ = p.getCameraImage(
                width=width, height=height,
                viewMatrix=view_matrix, projectionMatrix=proj_matrix,
                renderer=p.ER_TINY_RENDERER,
                physicsClientId=self._env.CLIENT
            )
            rgb_array = np.array(rgb, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
            return rgb_array
        except Exception:
            return np.zeros(self._size + (3,), dtype=np.uint8)

    def close(self):
        self._env.close()

    def set_total_steps(self, steps):
        self._total_steps = steps
