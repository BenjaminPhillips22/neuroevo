import torch
import numpy as np
import random
from torch.autograd import Variable
import ga_model


def evaluate(env_name, model, render=False,env_seed=2018,num_stack=4,cuda=False, max_eval=20000):
    if isinstance(model, ga_model.CompressedModel):
        model = ga_model.uncompress_model(model)
    if cuda:
        model.cuda()
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    from baselines.common.vec_env.vec_normalize import VecNormalize

    from envs import make_env
    from torch.autograd import Variable
    env = make_env(env_name, env_seed, 0, None, clip_rewards=False)
    env = DummyVecEnv([env])

    obs_shape = env.observation_space.shape
    obs_shape = (obs_shape[0] * num_stack, *obs_shape[1:])
    current_obs = torch.zeros(1, *obs_shape)
    states = torch.zeros(1, 512)
    masks = torch.zeros(1, 1)
        
    def update_current_obs(obs):
        shape_dim0 = env.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs
    render_func = env.envs[0].render


    if render: render_func('human')
    obs = env.reset()
    update_current_obs(obs)
    
    total_reward = 0.0
    frames = 0
    for fr in range(max_eval):
        frames += 1
        with torch.no_grad():
            current_obs_var = Variable(current_obs)
            states_var = Variable(states)
            masks_var = Variable(masks)
        if cuda:
            current_obs_var, states_var, masks_var = current_obs_var.cuda(), states_var.cuda(), masks_var.cuda()

        current_obs_var /= 255.0
        values = model(current_obs_var)[0]
        action = [np.argmax(values.cpu().data.numpy()[:env.action_space.n])]
        obs, reward, done, _ = env.step(action)


        total_reward += reward[0]
        masks.fill_(0.0 if done else 1.0)
    
        if current_obs.dim() == 4:
            current_obs *= masks.unsqueeze(2).unsqueeze(2)
        else:
            current_obs *= masks
        update_current_obs(obs)
    
        if render: render_func('human')
        if done:
            if _[0]['ale.lives']==0:
                break
    return total_reward, frames


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    #GA model
    model_path = '/home/leesy714/source/nn_landscape/trained_models/frames20000000_seed0/ga/FrostbiteNoFrameskip-v4_19.pt'
    seed = model_path[model_path.rfind('seed')+4:]
    seed = int(seed[:seed.find('/')])
    env_name = 'FrostbiteNoFrameskip-v4'
    origin = torch.load(model_path)
    if isinstance(origin, list):
        origin=origin[0]
    origin.cuda()
    result, frames = evaluate_model(env_name, origin, seed=2018,cuda=True, max_eval=5000)
    print(result,frames)
    

