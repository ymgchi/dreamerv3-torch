import sys
sys.path.insert(0, '/workspace/gym-pybullet-drones')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import numpy as np
import imageio
import ruamel.yaml as yaml
import models

from envs import drone_millisign_v4

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

# Use action_repeat=1 to match training config
env = drone_millisign_v4.PyBulletDroneMilliSignV4(
    task='follow',
    action_repeat=1,  # Match training!
    size=(64, 64),
    seed=42,
    reward_version='v21',
    stationary_target=True,
)

print('Creating model...')
wm = models.WorldModel(env.observation_space, env.action_space, 0, config)
task_behavior = models.ImagBehavior(config, wm)

agent_state = checkpoint['agent_state_dict']
wm_state = {k.replace('_wm.', '').replace('_orig_mod.', ''): v 
            for k, v in agent_state.items() if k.startswith('_wm.')}
tb_state = {k.replace('_task_behavior.', '').replace('_orig_mod.', ''): v 
            for k, v in agent_state.items() if k.startswith('_task_behavior.')}
wm.load_state_dict(wm_state)
task_behavior.load_state_dict(tb_state)
wm.eval()
task_behavior.eval()

print('Running evaluation (3 episodes)...')
all_frames = []
all_rewards = []

for ep in range(3):
    if ep > 0:
        env.close()
        env = drone_millisign_v4.PyBulletDroneMilliSignV4(
            task='follow',
            action_repeat=1,
            size=(64, 64),
            seed=42 + ep,
            reward_version='v21',
            stationary_target=True,
        )
    
    obs = env.reset()
    latent = None
    action = None
    total_reward = 0
    
    for step in range(300):  # Max 300 steps
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
        
        # Capture frame every 3rd step (10fps instead of 30fps to reduce file size)
        if step % 3 == 0:
            frame = env.render(mode='rgb_array')
            if frame is not None and len(frame.shape) == 3:
                all_frames.append(frame)
        
        if done:
            print(f'Episode {ep+1}: ended at step {step+1}, reward={total_reward:.1f}')
            break
    else:
        print(f'Episode {ep+1}: completed 300 steps, reward={total_reward:.1f}')
    
    all_rewards.append(total_reward)

print(f'\nAverage reward: {np.mean(all_rewards):.1f}')
print(f'Total frames: {len(all_frames)}')

if all_frames:
    imageio.mimsave('/workspace/logs/millisignv22/eval_v22_final.mp4', all_frames, fps=10)
    print('Saved video')

env.close()
