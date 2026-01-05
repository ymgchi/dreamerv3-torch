"""
Detailed evaluation of MilliSign v2 model with metrics
"""
import argparse
import pathlib
import sys
import numpy as np
import torch
import json
from datetime import datetime

sys.path.insert(0, "/workspace/gym-pybullet-drones")
sys.path.append(str(pathlib.Path(__file__).parent))

import ruamel.yaml as yaml
import models

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def to_attrdict(d):
    if isinstance(d, dict):
        return AttrDict({k: to_attrdict(v) for k, v in d.items()})
    return d


class MilliSignEnvV2Eval:
    """MilliSign v2 environment for evaluation."""

    def __init__(self, seed=42, person_speed=0.8):
        self._rng = np.random.RandomState(seed)
        self._person_speed = person_speed

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
        self._target_distance = 2.5
        self._target_height = 1.8

        self._current_stage = 3

    def _update_person_position(self):
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
        return self._person_pos.copy()

    def _get_tag_position(self):
        tag_pos = self._person_pos.copy()
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

    @property
    def observation_space(self):
        import gym
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

        extended_obs = np.concatenate([
            obs_flat,
            radar_obs,
            np.zeros(3, dtype=np.float32),
            np.array([self._target_distance, self._target_height, 0.0], dtype=np.float32),
            np.array([float(self._current_stage)], dtype=np.float32),
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

        detected = radar_obs[4] > 0.5
        reward += 0.5
        if not detected:
            reward -= 0.5
        else:
            dist_err = np.abs(radar_obs[0] - self._target_distance)
            reward += max(0, 2.0 - dist_err)
            height_err = np.abs(drone_pos[2] - self._target_height)
            reward += max(0, 1.0 - height_err)

        distance_error = radar_obs[0] - self._target_distance if radar_obs[4] > 0.5 else 0.0

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
            "distance_error": abs(np.linalg.norm(drone_pos - tag_pos) - self._target_distance),
        }

        return {"state": extended_obs.astype(np.float32), "is_first": False, "is_terminal": terminated}, reward, terminated or truncated, info

    def close(self):
        self._env.close()


def evaluate_scenario(wm, task_behavior, config, scenario_name, person_speed, episodes=10, max_steps=300):
    """Evaluate a specific scenario."""
    print(f"\n{'='*50}")
    print(f"Scenario: {scenario_name} (speed={person_speed})")
    print(f"{'='*50}")

    env = MilliSignEnvV2Eval(seed=42, person_speed=person_speed)

    all_rewards = []
    all_detection_rates = []
    all_distance_errors = []
    all_steps = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        step = 0
        latent = None
        action = None

        detected_steps = 0
        distance_errors = []

        while not done and step < max_steps:
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

            if info["detected"]:
                detected_steps += 1
            distance_errors.append(info["distance_error"])

        detection_rate = detected_steps / step * 100
        avg_dist_error = np.mean(distance_errors)

        all_rewards.append(episode_reward)
        all_detection_rates.append(detection_rate)
        all_distance_errors.append(avg_dist_error)
        all_steps.append(step)

        print(f"  Episode {ep+1:2d}: Reward={episode_reward:7.1f}, Detection={detection_rate:5.1f}%, DistErr={avg_dist_error:.2f}m")

    env.close()

    results = {
        "scenario": scenario_name,
        "person_speed": person_speed,
        "episodes": episodes,
        "avg_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "avg_detection_rate": float(np.mean(all_detection_rates)),
        "std_detection_rate": float(np.std(all_detection_rates)),
        "avg_distance_error": float(np.mean(all_distance_errors)),
        "std_distance_error": float(np.std(all_distance_errors)),
    }

    print(f"\n  Summary:")
    print(f"    Avg Reward:      {results['avg_reward']:.1f} +/- {results['std_reward']:.1f}")
    print(f"    Detection Rate:  {results['avg_detection_rate']:.1f}% +/- {results['std_detection_rate']:.1f}%")
    print(f"    Distance Error:  {results['avg_distance_error']:.2f}m +/- {results['std_distance_error']:.2f}m")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="/workspace/logdir/millisignv2")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=str, default="evaluation_results.json")
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

    # Load checkpoint
    checkpoint_path = logdir / "latest.pt"
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)

    # Create model
    print("Creating model...")
    env_temp = MilliSignEnvV2Eval()
    obs_space = env_temp.observation_space
    act_space = env_temp.action_space
    env_temp.close()

    wm = models.WorldModel(obs_space, act_space, 0, config)
    task_behavior = models.ImagBehavior(config, wm)

    # Load weights
    agent_state = checkpoint["agent_state_dict"]
    wm_state = {k.replace("_wm.", "").replace("_orig_mod.", ""): v
                for k, v in agent_state.items() if k.startswith("_wm.")}
    wm.load_state_dict(wm_state)

    tb_state = {k.replace("_task_behavior.", "").replace("_orig_mod.", ""): v
                for k, v in agent_state.items() if k.startswith("_task_behavior.")}
    task_behavior.load_state_dict(tb_state)

    wm.to(config.device)
    task_behavior.to(config.device)
    wm.eval()
    task_behavior.eval()

    # Define scenarios
    scenarios = [
        ("Stationary", 0.0),
        ("Slow", 0.3),
        ("Medium", 0.5),
        ("Fast", 0.8),
        ("Very Fast", 1.2),
    ]

    print("\n" + "="*60)
    print("MilliSign v2 Detailed Evaluation")
    print("="*60)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(checkpoint_path),
        "episodes_per_scenario": args.episodes,
        "scenarios": []
    }

    for scenario_name, speed in scenarios:
        results = evaluate_scenario(
            wm, task_behavior, config,
            scenario_name, speed,
            episodes=args.episodes
        )
        all_results["scenarios"].append(results)

    # Summary table
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    print(f"{'Scenario':<15} {'Reward':>10} {'Detection':>12} {'Dist Err':>10}")
    print("-"*50)
    for r in all_results["scenarios"]:
        print(f"{r['scenario']:<15} {r['avg_reward']:>10.1f} {r['avg_detection_rate']:>11.1f}% {r['avg_distance_error']:>9.2f}m")

    # Overall averages
    avg_reward = np.mean([r['avg_reward'] for r in all_results['scenarios']])
    avg_detection = np.mean([r['avg_detection_rate'] for r in all_results['scenarios']])
    avg_dist_err = np.mean([r['avg_distance_error'] for r in all_results['scenarios']])
    print("-"*50)
    print(f"{'AVERAGE':<15} {avg_reward:>10.1f} {avg_detection:>11.1f}% {avg_dist_err:>9.2f}m")

    all_results["overall"] = {
        "avg_reward": float(avg_reward),
        "avg_detection_rate": float(avg_detection),
        "avg_distance_error": float(avg_dist_err),
    }

    # Save results
    output_path = pathlib.Path(__file__).parent / args.output
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
