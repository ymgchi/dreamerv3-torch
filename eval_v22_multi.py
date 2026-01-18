import sys
sys.path.insert(0, '/workspace/gym-pybullet-drones')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import numpy as np
import imageio
import ruamel.yaml as yaml
import models
import networks

# Load env
from envs import drone_millisign_v4

# Load configs
configs = yaml.safe_load(open('configs.yaml'))

class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def to_attrdict(d):
    if isinstance(d, dict):
        return AttrDict({k: to_attrdict(v) for k, v in d.items()})
    return d

def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value

config = {}
recursive_update(config, configs['defaults'])
recursive_update(config, configs.get('millisignv22', {}))
config['device'] = 'cpu'
config = to_attrdict(config)
config.num_actions = 4

print('Loading checkpoint...')
checkpoint = torch.load('/workspace/logs/millisignv22/best_100000_266.8.pt', map_location='cpu', weights_only=False)
print(f'Checkpoint step: {checkpoint.get("step", "unknown")}')

# Create env
env = drone_millisign_v4.PyBulletDroneMilliSignV4(
    task='follow',
    action_repeat=8,
    size=(64, 64),
    seed=42,
    reward_version='v21',
    stationary_target=True,
)

print('Creating model...')
obs_space = env.observation_space
act_space = env.action_space

wm = models.WorldModel(obs_space, act_space, 0, config)
task_behavior = models.ImagBehavior(config, wm)

# Load weights
agent_state = checkpoint['agent_state_dict']

wm_state = {}
for k, v in agent_state.items():
    if k.startswith('_wm.'):
        new_key = k.replace('_wm.', '').replace('_orig_mod.', '')
        wm_state[new_key] = v
wm.load_state_dict(wm_state)

tb_state = {}
for k, v in agent_state.items():
    if k.startswith('_task_behavior.'):
        new_key = k.replace('_task_behavior.', '').replace('_orig_mod.', '')
        tb_state[new_key] = v
task_behavior.load_state_dict(tb_state)

wm.eval()
task_behavior.eval()

print('Running evaluation (5 episodes)...')
all_frames = []
all_rewards = []

for ep in range(5):
    env.close()
    env = drone_millisign_v4.PyBulletDroneMilliSignV4(
        task='follow',
        action_repeat=8,
        size=(64, 64),
        seed=42 + ep,
        reward_version='v21',
        stationary_target=True,
    )
    
    obs = env.reset()
    latent = None
    action = None
    total_reward = 0
    ep_frames = []
    
    for step in range(242):
        obs_proc = {'state': torch.tensor(obs['state']).unsqueeze(0).float()}
        obs_proc['is_first'] = torch.tensor([[step == 0]], dtype=torch.bool)
        obs_proc['is_terminal'] = torch.tensor([[False]], dtype=torch.bool)
        obs_proc['image'] = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        
        with torch.no_grad():
            obs_proc = wm.preprocess(obs_proc)
            embed = wm.encoder(obs_proc)
            latent, _ = wm.dynamics.obs_step(latent, action, embed, obs_proc['is_first'])
            feat = wm.dynamics.get_feat(latent)
            actor = task_behavior.actor(feat)
            action = actor.mode()
        
        action_np = action.squeeze(0).numpy()
        obs, reward, done, info = env.step(action_np)
        total_reward += reward
        
        frame = env.render(mode='rgb_array')
        if frame is not None and len(frame.shape) == 3:
            ep_frames.append(frame)
        
        if done:
            print(f'Episode {ep+1}: ended at step {step+1}, reward={total_reward:.1f}')
            break
    else:
        print(f'Episode {ep+1}: completed 242 steps, reward={total_reward:.1f}')
    
    all_frames.extend(ep_frames)
    all_rewards.append(total_reward)

print(f'\nAverage reward: {np.mean(all_rewards):.1f}')
print(f'Total frames: {len(all_frames)}')

if all_frames:
    imageio.mimsave('/workspace/logs/millisignv22/eval_v22_multi.mp4', all_frames, fps=30)
    print('Saved video to /workspace/logs/millisignv22/eval_v22_multi.mp4')

env.close()
