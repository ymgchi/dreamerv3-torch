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

env = drone_millisign_v4.PyBulletDroneMilliSignV4(
    task='follow',
    action_repeat=8,
    size=(64, 64),
    seed=42,
    reward_version='v21',
    stationary_target=True,
)

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

# Force device to CPU (don't add to encoder/decoder - they don't accept it)
config['device'] = 'cpu'
config = to_attrdict(config)
config.num_actions = 4

print('Loading checkpoint...')
checkpoint = torch.load('/workspace/logs/millisignv22/best_100000_266.8.pt', map_location='cpu', weights_only=False)
print(f'Checkpoint step: {checkpoint.get("step", "unknown")}')

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

print('Running evaluation...')
frames = []
obs = env.reset()
latent = None
action = None
total_reward = 0

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
        frames.append(frame)
    
    if step % 30 == 0:
        print(f'Step {step}: reward={total_reward:.1f}')
    
    if done:
        break

print(f'Final reward: {total_reward:.1f}')
print(f'Frames captured: {len(frames)}')

if frames:
    imageio.mimsave('/workspace/logs/millisignv22/eval_v22_100k.mp4', frames, fps=30)
    print('Saved video to /workspace/logs/millisignv22/eval_v22_100k.mp4')

env.close()
