"""
DreamerV3 wrapper for trajectory following drone task
"""
import gym
import numpy as np


class PyBulletDroneTrajectory:
    """DreamerV3 compatible wrapper for trajectory following task."""

    metadata = {}

    def __init__(self, task="circle", action_repeat=1, size=(64, 64), seed=0):
        """
        Initialize the trajectory following drone environment.

        Parameters
        ----------
        task : str
            Trajectory type: "circle", "figure8", "waypoints", "hover_moving"
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
            obs=ObservationType.KIN,  # Kinematic observations
            act=ActionType.RPM,  # Direct motor control
            pyb_freq=240,
            ctrl_freq=30,
        )

        self.reward_range = [-np.inf, np.inf]
        self._step_count = 0
        self._episode_step = 0

        # Trajectory parameters
        self._trajectory_radius = 0.5  # meters
        self._trajectory_height = 1.0  # meters
        self._trajectory_speed = 0.5   # radians per second (for circular)
        self._waypoint_threshold = 0.15  # meters to consider waypoint reached

        # Waypoints for waypoint-based tasks
        self._waypoints = None
        self._current_waypoint_idx = 0

        # Current target position
        self._target_pos = np.array([0.0, 0.0, 1.0])

    def _generate_trajectory(self):
        """Generate trajectory based on task type."""
        if self._task == "circle":
            # Target moves in a circle
            pass  # Computed dynamically in _update_target
        elif self._task == "figure8":
            # Target moves in a figure-8 pattern
            pass  # Computed dynamically in _update_target
        elif self._task == "waypoints":
            # Random waypoints
            n_waypoints = 4
            self._waypoints = []
            for _ in range(n_waypoints):
                wp = np.array([
                    self._rng.uniform(-0.5, 0.5),
                    self._rng.uniform(-0.5, 0.5),
                    self._rng.uniform(0.5, 1.5)
                ])
                self._waypoints.append(wp)
            self._current_waypoint_idx = 0
            self._target_pos = self._waypoints[0].copy()
        elif self._task == "hover_moving":
            # Simple moving hover target
            self._target_pos = np.array([0.0, 0.0, 1.0])
        else:
            # Default: static hover
            self._target_pos = np.array([0.0, 0.0, 1.0])

    def _update_target(self, drone_pos):
        """Update target position based on trajectory type and time."""
        t = self._episode_step / 30.0  # Time in seconds

        if self._task == "circle":
            # Circular trajectory
            angle = t * self._trajectory_speed
            self._target_pos = np.array([
                self._trajectory_radius * np.cos(angle),
                self._trajectory_radius * np.sin(angle),
                self._trajectory_height
            ])
        elif self._task == "figure8":
            # Figure-8 trajectory (lemniscate)
            angle = t * self._trajectory_speed
            scale = self._trajectory_radius
            self._target_pos = np.array([
                scale * np.sin(angle),
                scale * np.sin(angle) * np.cos(angle),
                self._trajectory_height
            ])
        elif self._task == "waypoints":
            # Check if reached current waypoint
            dist_to_waypoint = np.linalg.norm(drone_pos - self._target_pos)
            if dist_to_waypoint < self._waypoint_threshold:
                # Advance to next waypoint
                self._current_waypoint_idx = (self._current_waypoint_idx + 1) % len(self._waypoints)
                self._target_pos = self._waypoints[self._current_waypoint_idx].copy()
        elif self._task == "hover_moving":
            # Slowly moving hover target
            angle = t * 0.3
            self._target_pos = np.array([
                0.3 * np.sin(angle),
                0.3 * np.cos(angle),
                1.0 + 0.2 * np.sin(angle * 0.5)
            ])

    @property
    def observation_space(self):
        """Return observation space compatible with DreamerV3."""
        spaces = {}

        # Get the original observation space
        orig_space = self._env.observation_space
        if hasattr(orig_space, 'shape'):
            # Original state (12 dims) + target position (3 dims) + relative position (3 dims)
            flat_size = int(np.prod(orig_space.shape))
            state_size = flat_size + 6  # Add target pos and relative pos
            spaces["state"] = gym.spaces.Box(
                -np.inf, np.inf, (state_size,), dtype=np.float32
            )

        # Add image observation
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

        # Reshape action if needed
        if len(action.shape) == 1:
            action = action.reshape(self._env.action_space.shape)

        reward = 0.0
        done = False
        truncated = False
        info = {}

        for _ in range(self._action_repeat):
            obs, r, terminated, truncated, info = self._env.step(action)
            self._episode_step += 1

            # Get drone position
            drone_pos = obs.flatten()[:3]

            # Update target position
            self._update_target(drone_pos)

            # Compute custom reward based on distance to target
            dist_to_target = np.linalg.norm(drone_pos - self._target_pos)

            # Reward: high when close to target, penalize distance
            proximity_reward = max(0, 2 - dist_to_target ** 2)

            # Bonus for being very close
            if dist_to_target < 0.1:
                proximity_reward += 1.0

            # Penalty for being too far
            if dist_to_target > 1.0:
                proximity_reward -= 0.5

            # Add velocity alignment reward (optional)
            reward += proximity_reward

            done = terminated or truncated
            if done:
                break

        self._step_count += 1

        # Build observation dict
        obs_flat = np.array(obs, dtype=np.float32).flatten()
        drone_pos = obs_flat[:3]
        relative_pos = self._target_pos - drone_pos

        # Combine original obs with target info
        extended_obs = np.concatenate([
            obs_flat,
            self._target_pos,
            relative_pos
        ])

        obs_dict = {}
        obs_dict["state"] = extended_obs.astype(np.float32)
        obs_dict["image"] = self.render()
        obs_dict["is_terminal"] = terminated
        obs_dict["is_first"] = False

        info["discount"] = np.array(0.0 if terminated else 1.0, np.float32)
        info["target_pos"] = self._target_pos.copy()
        info["dist_to_target"] = np.linalg.norm(drone_pos - self._target_pos)

        return obs_dict, reward, done, info

    def reset(self):
        """Reset environment."""
        obs, info = self._env.reset()
        self._step_count = 0
        self._episode_step = 0
        self._current_waypoint_idx = 0

        # Generate new trajectory
        self._generate_trajectory()

        # Initialize target
        if self._task in ["circle", "figure8", "hover_moving"]:
            self._update_target(np.array([0, 0, 0]))

        obs_flat = np.array(obs, dtype=np.float32).flatten()
        drone_pos = obs_flat[:3]
        relative_pos = self._target_pos - drone_pos

        extended_obs = np.concatenate([
            obs_flat,
            self._target_pos,
            relative_pos
        ])

        obs_dict = {}
        obs_dict["state"] = extended_obs.astype(np.float32)
        obs_dict["image"] = self.render()
        obs_dict["is_terminal"] = False
        obs_dict["is_first"] = True

        return obs_dict

    def render(self, *args, **kwargs):
        """Render environment to RGB array."""
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")

        try:
            # PyBullet rendering
            import pybullet as p

            width, height = self._size

            # Camera follows target position somewhat
            cam_target = self._target_pos * 0.5  # Look between origin and target

            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[0, -2, 2],
                cameraTargetPosition=cam_target.tolist(),
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

            # Convert to numpy array (remove alpha channel)
            rgb_array = np.array(rgb, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
            return rgb_array

        except Exception as e:
            # Return placeholder if rendering fails
            return np.zeros(self._size + (3,), dtype=np.uint8)

    def close(self):
        """Close the environment."""
        self._env.close()
