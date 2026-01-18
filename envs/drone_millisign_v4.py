"""
DreamerV3 wrapper for MilliSign-based person following task (v4).

Improvements over v3:
1. State history buffer (past 5 frames of radar observations)
2. Predicted person direction in observation
3. Relative velocity between drone and person
4. Exploration reward after losing track
5. Direction-focused reward over distance
6. Jitter penalty for stable flight
7. Yaw alignment reward - drone faces target
8. Target heading angle in observation
9. [IMPROVED v5] Position-based reward instead of distance-based
   - Target is 2.5m BEHIND person in their movement direction
   - When person is stationary, target is on drone-to-person line
   - Observation includes XYZ error to target position
"""
import gym
import numpy as np
from collections import deque


class PyBulletDroneMilliSignV4:
    """DreamerV3 compatible wrapper for MilliSign-based person following (v4)."""

    metadata = {}

    # History buffer size
    HISTORY_LENGTH = 5

    def __init__(self, task="follow", action_repeat=1, size=(64, 64), seed=0, min_speed=None, reward_version="v12", stationary_target=False, goal_termination=False):
        from gym_pybullet_drones.envs.HoverAviary import HoverAviary
        from gym_pybullet_drones.utils.enums import ActionType, ObservationType

        self._action_repeat = action_repeat
        self._size = size
        self._task = task
        self._rng = np.random.RandomState(seed)
        self._reward_version = reward_version
        self._stationary_target = stationary_target
        self._goal_termination = goal_termination
        self._goal_distance = 0.5  # Goal reached when within 0.5m
        self._max_episode_steps = 900  # 30 seconds max (with goal_termination)

        # Parse task for speed mode (e.g., "follow_fast" -> min_speed=0.8)
        self._min_speed = min_speed
        if "_fast" in task:
            self._min_speed = 0.8
        elif "_veryfast" in task:
            self._min_speed = 1.0

        self._env = HoverAviary(
            gui=False,
            obs=ObservationType.KIN,
            act=ActionType.VEL,
            pyb_freq=240,
            ctrl_freq=30,
        )

        # Extend episode length for goal termination
        if self._goal_termination:
            self._env.EPISODE_LEN_SEC = 60  # 60 seconds max

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

        # Detection tracking
        self._detection_streak = 0
        self._last_detected = False
        self._lost_steps = 0

        # History buffers for temporal information
        self._radar_history = deque(maxlen=self.HISTORY_LENGTH)
        self._action_history = deque(maxlen=self.HISTORY_LENGTH)
        self._drone_pos_history = deque(maxlen=self.HISTORY_LENGTH)

        # Velocity history for v19 correlation reward
        self.VEL_HISTORY_LENGTH = 15  # ~0.5 sec at 30fps
        self._drone_vel_history = deque(maxlen=self.VEL_HISTORY_LENGTH)
        self._target_vel_history = deque(maxlen=self.VEL_HISTORY_LENGTH)
        # Previous positions for velocity
        self._prev_person_pos = None
        self._prev_drone_pos = None
        self._prev_action = None

        # Last known target direction (for exploration when lost)
        self._last_known_direction = np.array([1.0, 0.0, 0.0])
        self._search_angle = 0.0

        # Curriculum learning - Continuous (A案)
        # 線形補間で滑らかに難易度上昇
        self._curriculum_min_speed = 0.1   # 最初から少し動く（静止なし）
        self._curriculum_max_speed = 1.2   # 最高速度
        self._curriculum_ramp_steps = 400000  # 40万ステップで最高難度到達
        self._current_stage = 0  # 互換性のため残す

        self._movement_patterns = {
            "stationary": self._person_stationary,
            "walk_line": self._person_walk_line,
            "walk_circle": self._person_walk_circle,
            "walk_random": self._person_walk_random,
            "walk_zigzag": self._person_walk_zigzag,
            "walk_sudden": self._person_walk_sudden,
        }
        self._current_pattern = "stationary"

    def _get_curriculum_stage(self):
        """Return continuous progress (0.0 to 1.0) for observation."""
        progress = min(self._total_steps / self._curriculum_ramp_steps, 1.0)
        return progress

    def _get_person_speed_multiplier(self):
        """No curriculum - always use random challenging speed."""
        return self._rng.uniform(0.6, 1.2)  # Always fast

    # Movement patterns (same as v3)
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
        """Random walk with smooth direction changes."""
        speed_mult = self._get_person_speed_multiplier()
        dt = 1.0 / 30.0
        max_turn_rate = 1.0  # radians per second
        
        # Set new target direction every 60 steps
        if self._episode_step % 60 == 0:
            self._target_angle = self._rng.uniform(0, 2 * np.pi)
            self._target_speed = self._rng.uniform(0.3, 0.8) * speed_mult
        
        # Initialize if needed
        if not hasattr(self, "_current_angle"):
            self._current_angle = 0.0
            self._target_angle = 0.0
            self._target_speed = 0.5 * speed_mult
            self._current_speed = 0.5 * speed_mult
        
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
        speed_mult = self._get_person_speed_multiplier()
        if self._episode_step % 30 == 0:
            if self._rng.random() < 0.3:
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

    def _calculate_reward(self, drone_pos, tag_pos, radar_obs, action, obs_flat):
        """
        Reward function v12 - Potential-Based Reward Shaping (PBRS).

        Based on research:
        - "Revisiting Sparse Rewards for Goal-Reaching RL" (arXiv 2407.00324)
        - "Potential-Based Reward Shaping" theory guarantees policy invariance

        Potential function: Φ(s) = -position_error
        Shaping reward: F(s,s') = γΦ(s') - Φ(s)

        This is mathematically guaranteed to:
        1. Not change the optimal policy (no reward hacking)
        2. Accelerate learning by providing direction
        3. Automatically encode speed (faster approach = more reward per time)
        """
        reward = 0.0
        gamma = 0.99  # Discount factor

        # Get target position and calculate error
        target_pos = self._get_target_position(drone_pos, tag_pos)
        position_error = np.linalg.norm(drone_pos - target_pos)

        # === 1. POTENTIAL-BASED REWARD SHAPING ===
        # Φ(s) = -position_error (closer to target = higher potential)
        if self._prev_drone_pos is not None:
            prev_target_pos = self._get_target_position(self._prev_drone_pos, tag_pos)
            prev_error = np.linalg.norm(self._prev_drone_pos - prev_target_pos)

            # PBRS formula: F = γΦ(s') - Φ(s)
            phi_prev = -prev_error
            phi_curr = -position_error
            shaping_reward = gamma * phi_curr - phi_prev

            # Scale for effective learning
            reward += 10.0 * shaping_reward

        # === 2. TASK REWARD: Goal reaching ===
        if position_error < 0.5:
            reward += 20.0  # Strong bonus for reaching target
        elif position_error < 1.0:
            reward += 5.0   # Bonus for being close

        # === 3. MINIMUM TIME PENALTY ===
        # Encourages faster completion (from "Revisiting Sparse Rewards" paper)
        reward -= 0.5

        # === 4. Detection tracking (state update only, no reward) ===
        detected = radar_obs[4] > 0.5
        if detected:
            self._detection_streak += 1
            rel_pos = tag_pos - drone_pos
            if np.linalg.norm(rel_pos) > 0.1:
                self._last_known_direction = rel_pos / np.linalg.norm(rel_pos)
        else:
            self._detection_streak = 0
            self._lost_steps += 1
        self._last_detected = detected

        return reward

    def _calculate_reward_v14(self, drone_pos, tag_pos, radar_obs, action, obs_flat):
        """
        Reward function v14 - Goal-Forcing with Safety.

        Key changes from v12:
        1. Strong penalty for being far from goal (forces convergence)
        2. Detection is REQUIRED for full reward (prevents blind flying)
        3. Velocity penalty for safety
        4. Exponential reward near goal
        """
        reward = 0.0

        # Get target position and calculate error
        target_pos = self._get_target_position(drone_pos, tag_pos)
        position_error = np.linalg.norm(drone_pos - target_pos)

        # Detection status
        detected = radar_obs[4] > 0.5

        # === 1. POSITION REWARD (Exponential near goal) ===
        if position_error < 0.5:
            # Goal reached! Big reward
            reward += 15.0
        elif position_error < 1.0:
            # Very close - good reward
            reward += 8.0 * (1.0 - position_error)
        elif position_error < 2.0:
            # Close - moderate reward
            reward += 2.0 * (1.0 - (position_error - 1.0))
        else:
            # Far - PENALTY (this forces goal seeking)
            reward -= 1.0 * (position_error - 2.0)
            reward = max(reward, -5.0)  # Cap penalty

        # === 2. DETECTION MULTIPLIER (Detection required!) ===
        if not detected:
            # Without detection, severely reduce position reward
            reward *= 0.2
            # Additional penalty for lost tracking
            reward -= 1.5

        # === 3. APPROACH REWARD (PBRS component) ===
        if self._prev_drone_pos is not None:
            prev_target_pos = self._get_target_position(self._prev_drone_pos, tag_pos)
            prev_error = np.linalg.norm(self._prev_drone_pos - prev_target_pos)
            error_reduction = prev_error - position_error

            # Reward for getting closer (only if detected)
            if detected:
                reward += 5.0 * error_reduction

        # === 4. VELOCITY PENALTY (Safety) ===
        if self._prev_drone_pos is not None:
            drone_vel = (drone_pos - self._prev_drone_pos) * 30.0
            speed = np.linalg.norm(drone_vel)
            if speed > 3.0:
                reward -= 0.5 * (speed - 3.0)  # Penalty for excessive speed

        # === 5. STAGNATION PENALTY ===
        if self._prev_drone_pos is not None and position_error > 1.0:
            movement = np.linalg.norm(drone_pos[:2] - self._prev_drone_pos[:2])
            if movement < 0.02:
                reward -= 0.5  # Penalty for not moving when far from goal

        # === 6. TIME PENALTY (Minimum time) ===
        reward -= 0.3

        # Update detection tracking
        if detected:
            self._detection_streak += 1
            self._lost_steps = 0
            rel_pos = tag_pos - drone_pos
            if np.linalg.norm(rel_pos) > 0.1:
                self._last_known_direction = rel_pos / np.linalg.norm(rel_pos)
        else:
            self._detection_streak = 0
            self._lost_steps += 1
        self._last_detected = detected

        return reward


    def _calculate_reward_v15(self, drone_pos, tag_pos, radar_obs, action, obs_flat):
        """
        Reward v15 - Simple design
        
        Principles:
        1. Maximum 3 reward terms
        2. Detection failure = no bonus (not penalty)
        3. Unified scale (-1 to +1 range)
        """
        target_pos = self._get_target_position(drone_pos, tag_pos)
        distance = np.linalg.norm(drone_pos - target_pos)
        detected = radar_obs[4] > 0.5
        
        # === 1. Distance reward (main) ===
        # Positive within 2m, negative beyond
        reward = 1.0 - (distance / 2.0)  # 0m: +1.0, 2m: 0, 4m: -1.0
        reward = np.clip(reward, -1.0, 1.0)
        
        # === 2. Detection bonus ===
        if detected:
            reward += 0.3
        
        # === 3. Success bonus (sparse) ===
        if distance < 0.5 and detected:
            reward += 0.5
        
        return reward

    def _calculate_reward_v17(self, drone_pos, tag_pos, radar_obs, action, obs_flat):
        """
        Reward v17 - Progress-based reward (PBRS-inspired)
        
        Based on research papers:
        - Progress reward: reward improvement in distance, not absolute distance
        - Small distance penalty to encourage staying close
        - No complex bonuses - keep it simple and learnable
        """
        target_pos = self._get_target_position(drone_pos, tag_pos)
        distance = np.linalg.norm(drone_pos - target_pos)
        
        # Initialize prev_distance if not set
        if not hasattr(self, "_prev_distance") or self._prev_distance is None:
            self._prev_distance = distance
        
        # === 1. Progress reward (main) ===
        # Positive when getting closer, negative when getting farther
        progress = self._prev_distance - distance
        reward = 10.0 * progress  # Scale up to make signal clearer
        
        # === 2. Small distance penalty ===
        # Encourages staying close, but dominated by progress
        reward -= 0.1 * distance
        
        # Update prev_distance for next step
        self._prev_distance = distance
        
        return reward

    def _calculate_reward_v18(self, drone_pos, tag_pos, radar_obs, action, obs_flat):
        """
        Reward v18 - Approach speed based reward
        
        Key insight: Reward drone's OWN action, not distance change
        - Use drone's velocity toward target (approach speed)
        - This decouples reward from target's movement
        """
        target_pos = self._get_target_position(drone_pos, tag_pos)
        distance = np.linalg.norm(drone_pos - target_pos)
        
        # Direction to target (normalized)
        to_target = target_pos - drone_pos
        if np.linalg.norm(to_target) > 0.01:
            to_target_norm = to_target / np.linalg.norm(to_target)
        else:
            to_target_norm = np.zeros(3)
        
        # Drone velocity (using prev_drone_pos)
        if self._prev_drone_pos is not None:
            drone_vel = (drone_pos - self._prev_drone_pos) * 30.0  # Scale by fps
        else:
            drone_vel = np.zeros(3)
        
        # === 1. Approach speed reward (main) ===
        # Dot product of velocity and direction to target
        approach_speed = np.dot(drone_vel, to_target_norm)
        reward = 5.0 * approach_speed  # Positive when moving toward target
        
        # === 2. Small distance bonus when close ===
        # Encourage staying close once reached
        if distance < 1.5:
            reward += 0.5 * (1.5 - distance)  # Max +0.75 at 0m
        
        # === 3. Small penalty for being far ===
        if distance > 3.0:
            reward -= 0.1 * (distance - 3.0)  # Penalty beyond 3m
        
        return reward

    def _calculate_reward_v19(self, drone_pos, tag_pos, radar_obs, action, obs_flat):
        """
        Reward v19 - Full tracking reward with correlation
        
        Components:
        1. xy/z separated approach reward (prioritize horizontal pursuit)
        2. Relative velocity reward (closing speed considering target movement)
        3. Cross-correlation reward (lag-tolerant pattern matching)
        """
        target_pos = self._get_target_position(drone_pos, tag_pos)
        distance = np.linalg.norm(drone_pos - target_pos)
        
        # Direction to target
        to_target = target_pos - drone_pos
        to_target_xy = to_target[:2]
        to_target_z = to_target[2]
        
        if np.linalg.norm(to_target_xy) > 0.01:
            to_target_xy_norm = to_target_xy / np.linalg.norm(to_target_xy)
        else:
            to_target_xy_norm = np.zeros(2)
        
        to_target_z_sign = np.sign(to_target_z) if abs(to_target_z) > 0.01 else 0
        
        # Drone velocity
        if self._prev_drone_pos is not None:
            drone_vel = (drone_pos - self._prev_drone_pos) * 30.0
        else:
            drone_vel = np.zeros(3)
        
        drone_vel_xy = drone_vel[:2]
        drone_vel_z = drone_vel[2]
        
        # Target velocity
        target_vel = self._person_vel.copy()
        target_vel_xy = target_vel[:2]
        
        # === 1. xy/z separated approach reward ===
        xy_approach = np.dot(drone_vel_xy, to_target_xy_norm)
        z_approach = drone_vel_z * to_target_z_sign
        reward_approach = 6.0 * xy_approach + 2.0 * z_approach  # Prioritize xy
        
        # === 2. Relative velocity reward (closing speed) ===
        relative_vel_xy = drone_vel_xy - target_vel_xy
        if np.linalg.norm(to_target_xy) > 0.01:
            closing_speed = np.dot(relative_vel_xy, to_target_xy_norm)
            reward_closing = 3.0 * closing_speed
        else:
            reward_closing = 0.0
        
        # === 3. Cross-correlation reward (lag-tolerant) ===
        # Update velocity history
        self._drone_vel_history.append(drone_vel_xy.copy())
        self._target_vel_history.append(target_vel_xy.copy())
        
        reward_correlation = 0.0
        if len(self._drone_vel_history) >= 5:  # Need enough history
            drone_hist = np.array(list(self._drone_vel_history))
            target_hist = np.array(list(self._target_vel_history))
            
            # Compute correlation at different lags
            max_lag = min(5, len(drone_hist) - 1)
            best_corr = -1.0
            
            for lag in range(max_lag + 1):
                if lag == 0:
                    d = drone_hist
                    t = target_hist
                else:
                    d = drone_hist[lag:]
                    t = target_hist[:-lag]
                
                # Flatten and compute correlation
                d_flat = d.flatten()
                t_flat = t.flatten()
                
                d_norm = np.linalg.norm(d_flat)
                t_norm = np.linalg.norm(t_flat)
                
                if d_norm > 0.01 and t_norm > 0.01:
                    corr = np.dot(d_flat, t_flat) / (d_norm * t_norm)
                    best_corr = max(best_corr, corr)
            
            reward_correlation = 2.0 * best_corr  # Scale correlation reward
        
        # === 4. Distance bonus/penalty ===
        if distance < 1.5:
            reward_distance = 0.3 * (1.5 - distance)
        elif distance > 3.0:
            reward_distance = -0.1 * (distance - 3.0)
        else:
            reward_distance = 0.0
        
        total_reward = reward_approach + reward_closing + reward_correlation + reward_distance
        return total_reward

    def _calculate_reward_v20(self, drone_pos, tag_pos, radar_obs, action, obs_flat):
        """
        Reward v20 - v18 + horizontal boost (minimal change)
        
        Based on v18 which worked (311.9 at 140k)
        Only change: extra reward for horizontal (xy) approach
        """
        target_pos = self._get_target_position(drone_pos, tag_pos)
        distance = np.linalg.norm(drone_pos - target_pos)
        
        # Direction to target (3D)
        to_target = target_pos - drone_pos
        if np.linalg.norm(to_target) > 0.01:
            to_target_norm = to_target / np.linalg.norm(to_target)
        else:
            to_target_norm = np.zeros(3)
        
        # Direction to target (xy only)
        to_target_xy = to_target[:2]
        if np.linalg.norm(to_target_xy) > 0.01:
            to_target_xy_norm = to_target_xy / np.linalg.norm(to_target_xy)
        else:
            to_target_xy_norm = np.zeros(2)
        
        # Drone velocity
        if self._prev_drone_pos is not None:
            drone_vel = (drone_pos - self._prev_drone_pos) * 30.0
        else:
            drone_vel = np.zeros(3)
        
        drone_vel_xy = drone_vel[:2]
        
        # === 1. Approach speed (v18 base) ===
        approach_speed = np.dot(drone_vel, to_target_norm)
        reward = 5.0 * approach_speed
        
        # === 2. Horizontal boost (NEW) ===
        xy_approach = np.dot(drone_vel_xy, to_target_xy_norm)
        reward += 3.0 * xy_approach
        
        # === 3. Distance bonus (v18 same) ===
        if distance < 1.5:
            reward += 0.5 * (1.5 - distance)
        
        if distance > 3.0:
            reward -= 0.1 * (distance - 3.0)
        
        return reward

    def _calculate_reward_v21(self, drone_pos, tag_pos, radar_obs, action, obs_flat):
        """
        Reward v21 - v20 + strong distance penalty
        
        Fix: drone was floating because no penalty for being far
        Add continuous distance penalty to create urgency
        """
        target_pos = self._get_target_position(drone_pos, tag_pos)
        distance = np.linalg.norm(drone_pos - target_pos)
        
        # Direction to target (3D)
        to_target = target_pos - drone_pos
        if np.linalg.norm(to_target) > 0.01:
            to_target_norm = to_target / np.linalg.norm(to_target)
        else:
            to_target_norm = np.zeros(3)
        
        # Direction to target (xy only)
        to_target_xy = to_target[:2]
        if np.linalg.norm(to_target_xy) > 0.01:
            to_target_xy_norm = to_target_xy / np.linalg.norm(to_target_xy)
        else:
            to_target_xy_norm = np.zeros(2)
        
        # Drone velocity
        if self._prev_drone_pos is not None:
            drone_vel = (drone_pos - self._prev_drone_pos) * 30.0
        else:
            drone_vel = np.zeros(3)
        
        drone_vel_xy = drone_vel[:2]
        
        # === 1. Approach speed (v20 same) ===
        approach_speed = np.dot(drone_vel, to_target_norm)
        reward = 5.0 * approach_speed
        
        # === 2. Horizontal boost (v20 same) ===
        xy_approach = np.dot(drone_vel_xy, to_target_xy_norm)
        reward += 3.0 * xy_approach
        
        # === 3. STRONG distance penalty (NEW) ===
        # Every step, penalize based on distance
        # At 3m: -1.5/step, at 1m: -0.5/step
        reward -= 0.5 * distance
        
        # === 4. Close bonus (keep from v20) ===
        if distance < 1.0:
            reward += 1.0 * (1.0 - distance)  # Stronger bonus when very close
        
        return reward

    def _get_history_observation(self):
        """Get flattened history of radar observations."""
        # Pad with zeros if not enough history
        history = list(self._radar_history)
        while len(history) < self.HISTORY_LENGTH:
            history.insert(0, np.zeros(5, dtype=np.float32))
        return np.concatenate(history)

    def _get_predicted_person_direction(self):
        """Predict person's future direction based on velocity."""
        if np.linalg.norm(self._person_vel[:2]) > 0.05:
            direction = self._person_vel[:2] / np.linalg.norm(self._person_vel[:2])
            return direction.astype(np.float32)
        return np.zeros(2, dtype=np.float32)

    def _get_relative_velocity(self, drone_pos):
        """Get relative velocity between drone and person."""
        if self._prev_drone_pos is not None and self._prev_person_pos is not None:
            drone_vel = (drone_pos - self._prev_drone_pos) * 30.0
            person_vel = (self._person_pos - self._prev_person_pos) * 30.0
            rel_vel = drone_vel - person_vel
            return rel_vel.astype(np.float32)
        return np.zeros(3, dtype=np.float32)

    def _get_drone_yaw(self, obs_flat):
        """Extract drone's yaw angle from observation.

        PyBullet KIN observation structure (first 12 values per drone):
        [0:3] - position (x, y, z)
        [3:7] - quaternion (qx, qy, qz, qw)
        [7:10] - roll, pitch, yaw (Euler angles)
        [10:13] - linear velocity
        [13:16] - angular velocity
        """
        # Yaw is at index 9 (after pos[3], quat[4], roll[1], pitch[1])
        if len(obs_flat) >= 10:
            return obs_flat[9]
        return 0.0

    def _get_heading_to_target(self, drone_pos, tag_pos, drone_yaw):
        """Get the heading error - angle between drone's facing direction and target.

        Returns:
        - heading_error: angle difference (normalized to [-pi, pi])
        - target_angle: absolute angle to target from world frame
        """
        # Angle to target in world frame
        rel_pos = tag_pos[:2] - drone_pos[:2]
        target_angle = np.arctan2(rel_pos[1], rel_pos[0])

        # Heading error (how much drone needs to rotate to face target)
        heading_error = target_angle - drone_yaw

        # Normalize to [-pi, pi]
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi

        return heading_error, target_angle

    def _get_target_position(self, drone_pos, tag_pos):
        """Calculate the target XYZ position for the drone.

        Target is 2.5m behind the person in their movement direction.
        If person is stationary, target is 2.5m in the direction from drone to person.

        Returns:
        - target_pos: 3D target position for the drone
        """
        # Check if person is moving
        person_speed = np.linalg.norm(self._person_vel[:2])

        if person_speed > 0.05:
            # Person is moving - target is behind them
            person_dir = self._person_vel[:2] / person_speed
            target_xy = tag_pos[:2] - self._target_distance * person_dir
        else:
            # Person is stationary - target is on the line from drone to person
            rel_pos = tag_pos[:2] - drone_pos[:2]
            dist_to_tag = np.linalg.norm(rel_pos)
            if dist_to_tag > 0.1:
                direction = rel_pos / dist_to_tag
                target_xy = tag_pos[:2] - self._target_distance * direction
            else:
                # Drone is very close to tag, use last known direction
                target_xy = tag_pos[:2] - self._target_distance * self._last_known_direction[:2]

        target_pos = np.array([target_xy[0], target_xy[1], self._target_height], dtype=np.float32)
        return target_pos

    @property
    def observation_space(self):
        spaces = {}
        orig_space = self._env.observation_space
        if hasattr(orig_space, 'shape'):
            flat_size = int(np.prod(orig_space.shape))
        else:
            flat_size = 12

        # Extended state:
        # - drone(72 from PyBullet)
        # - radar(5)
        # - radar_history(5 * 5 = 25)
        # - person_velocity(3)
        # - predicted_direction(2)
        # - relative_velocity(3)
        # - target_info(3)
        # - heading_info(2) - [heading_error, target_angle] [NEW]
        # - curriculum(1)
        # - streak(1)
        # - lost_steps_normalized(1)
        state_size = flat_size + 5 + 25 + 3 + 2 + 3 + 3 + 2 + 1 + 1 + 1

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

        action_flat = action.flatten()
        reward = 0.0
        done = False
        info = {}

        for _ in range(self._action_repeat):
            obs, r, terminated, truncated, env_info = self._env.step(action)
            self._episode_step += 1
            self._total_steps += 1

            obs_flat_inner = np.array(obs, dtype=np.float32).flatten()
            drone_pos = obs_flat_inner[:3]
            tag_pos = self._get_tag_position()
            radar_obs = self._simulate_radar_detection(drone_pos, tag_pos)

            # Store in history
            self._radar_history.append(radar_obs.copy())
            self._drone_pos_history.append(drone_pos.copy())

            # Select reward function based on version
            if self._reward_version == "v21":
                step_reward = self._calculate_reward_v21(drone_pos, tag_pos, radar_obs, action_flat, obs_flat_inner)
            elif self._reward_version == "v20":
                step_reward = self._calculate_reward_v20(drone_pos, tag_pos, radar_obs, action_flat, obs_flat_inner)
            elif self._reward_version == "v19":
                step_reward = self._calculate_reward_v19(drone_pos, tag_pos, radar_obs, action_flat, obs_flat_inner)
            elif self._reward_version == "v18":
                step_reward = self._calculate_reward_v18(drone_pos, tag_pos, radar_obs, action_flat, obs_flat_inner)
            elif self._reward_version == "v17":
                step_reward = self._calculate_reward_v17(drone_pos, tag_pos, radar_obs, action_flat, obs_flat_inner)
            elif self._reward_version == "v15":
                step_reward = self._calculate_reward_v15(drone_pos, tag_pos, radar_obs, action_flat, obs_flat_inner)
            elif self._reward_version == "v14":
                step_reward = self._calculate_reward_v14(drone_pos, tag_pos, radar_obs, action_flat, obs_flat_inner)
            else:
                step_reward = self._calculate_reward(drone_pos, tag_pos, radar_obs, action_flat, obs_flat_inner)
            reward += step_reward

            # Update previous positions
            self._prev_drone_pos = drone_pos.copy()
            self._prev_person_pos = self._person_pos.copy()
            self._prev_action = action_flat.copy()

            done = terminated or truncated
            if drone_pos[2] < 0.1:
                done = True
                reward -= 5.0
            if np.any(np.abs(drone_pos[:2]) > 7.0):
                done = True
                reward -= 5.0

            # Goal termination check (v23)
            if self._goal_termination:
                tag_pos = self._get_tag_position()
                target_pos = self._get_target_position(drone_pos, tag_pos)
                dist_to_target = np.linalg.norm(drone_pos - target_pos)
                if dist_to_target < self._goal_distance:
                    done = True
                    reward += 20.0  # Success bonus
                elif self._episode_step >= self._max_episode_steps:
                    done = True
                    reward -= 5.0  # Timeout penalty

            if done:
                break

        self._step_count += 1

        obs_flat = np.array(obs, dtype=np.float32).flatten()
        drone_pos = obs_flat[:3]

        if self._prev_person_pos is not None:
            person_vel_est = (self._person_pos - self._prev_person_pos) * 30.0
        else:
            person_vel_est = np.zeros(3)

        # [IMPROVED v5] Include target position error instead of just distance error
        target_pos = self._get_target_position(drone_pos, tag_pos)
        target_error = target_pos - drone_pos  # Vector from drone to target
        target_info = np.array([
            target_error[0],  # X error to target
            target_error[1],  # Y error to target
            target_error[2],  # Z error to target
        ], dtype=np.float32)

        curriculum_info = np.array([self._get_curriculum_stage()], dtype=np.float32)  # Already 0.0-1.0
        streak_info = np.array([min(self._detection_streak / 60.0, 1.0)], dtype=np.float32)
        lost_info = np.array([min(self._lost_steps / 30.0, 1.0)], dtype=np.float32)

        # [NEW] Heading info - angle error and target angle
        drone_yaw = self._get_drone_yaw(obs_flat)
        heading_error, target_angle = self._get_heading_to_target(drone_pos, tag_pos, drone_yaw)
        heading_info = np.array([
            heading_error / np.pi,  # Normalized to [-1, 1]
            target_angle / np.pi,   # Normalized to [-1, 1]
        ], dtype=np.float32)

        # Build extended observation
        extended_obs = np.concatenate([
            obs_flat,                                  # 72
            radar_obs,                                 # 5
            self._get_history_observation(),          # 25
            person_vel_est.astype(np.float32),        # 3
            self._get_predicted_person_direction(),   # 2
            self._get_relative_velocity(drone_pos),   # 3
            target_info,                              # 3
            heading_info,                             # 2 [NEW]
            curriculum_info,                          # 1
            streak_info,                              # 1
            lost_info,                                # 1
        ])

        obs_dict = {
            "state": extended_obs.astype(np.float32),
            "image": self.render(),
            "is_terminal": terminated,
            "is_first": False,
        }

        info["discount"] = np.array(0.0 if terminated else 1.0, np.float32)
        info["tag_pos"] = tag_pos.copy()
        info["target_pos"] = target_pos.copy()  # [NEW v5] Target position
        info["position_error"] = np.linalg.norm(drone_pos - target_pos)  # [NEW v5]
        info["radar_detection"] = radar_obs.copy()
        info["distance_to_tag"] = np.linalg.norm(drone_pos - tag_pos)
        info["curriculum_stage"] = self._get_curriculum_stage()
        info["detection_streak"] = self._detection_streak
        info["lost_steps"] = self._lost_steps
        info["heading_error"] = heading_error
        info["drone_yaw"] = drone_yaw

        return obs_dict, reward, done, info

    def reset(self):
        obs, info = self._env.reset()
        self._step_count = 0
        self._episode_step = 0
        self._prev_person_pos = None
        self._prev_drone_pos = None
        self._prev_action = None
        self._detection_streak = 0
        self._last_detected = False
        self._lost_steps = 0
        self._prev_distance = None  # for v17 reward
        self._search_angle = 0.0

        # Clear history buffers
        self._radar_history.clear()
        self._action_history.clear()
        self._drone_pos_history.clear()
        self._drone_vel_history.clear()
        self._target_vel_history.clear()

        # No curriculum - always use challenging conditions
        self._person_pos = np.array([
            self._rng.uniform(1.5, 3.5),
            self._rng.uniform(-1.5, 1.5),
            0.0
        ])
        if self._stationary_target:
            self._current_pattern = "stationary"
        else:
            self._current_pattern = self._rng.choice(["walk_random", "walk_zigzag", "walk_sudden"])
        self._person_vel = np.zeros(3)
        self._last_known_direction = np.array([1.0, 0.0, 0.0])

        obs_flat = np.array(obs, dtype=np.float32).flatten()
        drone_pos = obs_flat[:3]
        tag_pos = self._get_tag_position()
        radar_obs = self._simulate_radar_detection(drone_pos, tag_pos)

        # Initialize history with current observation
        for _ in range(self.HISTORY_LENGTH):
            self._radar_history.append(radar_obs.copy())

        person_vel_est = np.zeros(3, dtype=np.float32)
        # [IMPROVED v5] Include target position error
        target_pos = self._get_target_position(drone_pos, tag_pos)
        target_error = target_pos - drone_pos
        target_info = np.array([target_error[0], target_error[1], target_error[2]], dtype=np.float32)
        curriculum_info = np.array([1.0], dtype=np.float32)  # Always max difficulty
        streak_info = np.array([0.0], dtype=np.float32)
        lost_info = np.array([0.0], dtype=np.float32)

        # [NEW] Heading info
        drone_yaw = self._get_drone_yaw(obs_flat)
        heading_error, target_angle = self._get_heading_to_target(drone_pos, tag_pos, drone_yaw)
        heading_info = np.array([
            heading_error / np.pi,
            target_angle / np.pi,
        ], dtype=np.float32)

        extended_obs = np.concatenate([
            obs_flat,
            radar_obs,
            self._get_history_observation(),
            person_vel_est,
            self._get_predicted_person_direction(),
            np.zeros(3, dtype=np.float32),  # relative velocity
            target_info,
            heading_info,  # [NEW]
            curriculum_info,
            streak_info,
            lost_info,
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
