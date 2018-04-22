import torch
import numpy as np
import random
from torch.autograd import Variable
import ga_model


def evaluate(env_name, models, render=False,env_seed=2018,num_stack=4,cuda=False, max_eval=20000):
    if isinstance(models[0], ga_model.CompressedModel):
        models = [ga_model.uncompress_model(model)  for model in models]
    if cuda:
        for model in models:
            model.cuda()

    from envs import make_env
    from torch.autograd import Variable
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    
    num_proc = len(models)
    env = [ make_env(env_name, env_seed, i, None, clip_rewards=False) for i in range(num_proc)]
    env = SubprocVecEnv(env)

    obs_shape = env.observation_space.shape
    obs_shape = (obs_shape[0] * num_stack, *obs_shape[1:])
    current_obs = torch.zeros(num_proc, *obs_shape)
        
    def update_current_obs(obs):
        shape_dim0 = env.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs
    render_func = env.render


    if render: render_func('human')
    obs = env.reset()
    update_current_obs(obs)
    
    total_reward = np.zeros(num_proc)
    masks = np.ones(num_proc)
    frames = 0
    for fr in range(max_eval):
        frames += num_proc 
        actions=[]
        for model in models:
            with torch.no_grad():
                current_obs_var = Variable(current_obs)
                if cuda:
                    current_obs_var = current_obs_var.cuda()
    
                current_obs_var /= 255.0
                values = model(current_obs_var)[0]
                action = np.argmax(values.cpu().data.numpy()[:env.action_space.n])
                actions.append(action)
        obs, reward, done, _ = env.step(actions)
        for i, d in enumerate(done):
            masks[i] = 0 if masks[i]==0 or _[i]['ale.lives']==0 else 1

        total_reward += reward * masks
    
        update_current_obs(obs)
    
        if render: render_func('human')
        if np.sum(masks) == 0:
            break
    env.close()
    return total_reward, frames


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import time
    #GA model
    origin = ga_model.Model(3)
    origin2 = ga_model.Model(5)
    origins=[origin, origin2]
    env_name = 'PongNoFrameskip-v4'
    start = time.time()
    result, frames = evaluate(env_name, origins, env_seed=2018,cuda=True, max_eval=5000)
    print(result,frames, time.time()-start)
    

