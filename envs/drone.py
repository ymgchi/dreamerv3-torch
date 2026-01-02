"""
DreamerV3 wrapper for gym-pybullet-drones
"""
import gym
import numpy as np


class PyBulletDrone:
    """DreamerV3 compatible wrapper for gym-pybullet-drones."""

    metadata = {}

    def __init__(self, task="hover", action_repeat=1, size=(64, 64), seed=0):
        """
        Initialize the drone environment.

        Parameters
        ----------
        task : str
            Task name: "hover" (single drone hover)
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

    @property
    def observation_space(self):
        """Return observation space compatible with DreamerV3."""
        spaces = {}

        # Get the original observation space
        orig_space = self._env.observation_space
        if hasattr(orig_space, 'shape'):
            # Box space - flatten to single key (1D shape for DreamerV3)
            flat_size = int(np.prod(orig_space.shape))
            spaces["state"] = gym.spaces.Box(
                -np.inf, np.inf, (flat_size,), dtype=np.float32
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
            reward += r
            done = terminated or truncated
            if done:
                break

        self._step_count += 1

        # Build observation dict
        obs_dict = {}
        obs_dict["state"] = np.array(obs, dtype=np.float32).flatten()
        obs_dict["image"] = self.render()
        obs_dict["is_terminal"] = terminated
        obs_dict["is_first"] = False

        info["discount"] = np.array(0.0 if terminated else 1.0, np.float32)

        return obs_dict, reward, done, info

    def reset(self):
        """Reset environment."""
        obs, info = self._env.reset()
        self._step_count = 0

        obs_dict = {}
        obs_dict["state"] = np.array(obs, dtype=np.float32).flatten()
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
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[0, -2, 2],
                cameraTargetPosition=[0, 0, 0.5],
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
