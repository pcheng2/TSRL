import numpy as np
import d4rl
import gym
from sklearn.model_selection import train_test_split
import collections
import os 

def sequence_dataset(env, dataset=None, **kwargs):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N):
        done_bool = dataset['terminals'][i]
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in dataset.keys():
            # print(k)
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1
    return data_

def traj_to_array(trajs):
    for traj_num in range(len(trajs)):
        if traj_num == 0:
            states = trajs[traj_num]['observations']
            actions = trajs[traj_num]['actions']
            next_observations = trajs[traj_num]['next_observations']
            rewards = trajs[traj_num]['rewards']
            terminals = trajs[traj_num]['terminals']
            
            
        else:
            states = np.concatenate((states, trajs[traj_num]['observations']), 0)
            actions = np.concatenate((actions, trajs[traj_num]['actions']), 0)
            next_observations = np.concatenate((next_observations, trajs[traj_num]['next_observations']), 0)
            rewards = np.squeeze(np.concatenate((rewards, trajs[traj_num]['rewards']), 0))
            terminals = np.squeeze(np.concatenate((terminals, trajs[traj_num]['terminals']), 0))

    return states, actions, next_observations, rewards, terminals

envs_1M = ['hopper-random-v2', 'hopper-medium-v2','hopper-expert-v2', 
        'halfcheetah-random-v2','halfcheetah-medium-v2', 'halfcheetah-expert-v2', 
        'walker2d-random-v2','walker2d-medium-v2','walker2d-expert-v2' ]
ratio_1M = [100, 20, 10]

envs_ant_maze = ['antmaze-umaze-v0', 'antmaze-umaze-diverse-v0','antmaze-medium-diverse-v0', 
        'antmaze-medium-play-v0','antmaze-large-diverse-v0', 'antmaze-large-play-v0']
ratio_ant_maze = [200]


envs_2M = ['hopper-medium-expert-v2','halfcheetah-medium-expert-v2','walker2d-medium-expert-v2']
ratio_2M = [200, 40, 20]

# envs_adroit = ['pen-cloned-v1', 'hammer-cloned-v1','door-cloned-v1', 
#         'relocate-cloned-v1','pen-expert-v1', 'hammer-expert-v1', 
#         'door-expert-v1','relocate-expert-v1']

envs_adroit = ['pen-cloned-v0', 'hammer-cloned-v0','door-cloned-v0', 
        'relocate-cloned-v0','pen-expert-v0', 'hammer-expert-v0', 
        'door-expert-v0','relocate-expert-v0']

# envs_adroit = ['pen-expert-v1', 'hammer-expert-v1', 'door-expert-v1','relocate-expert-v1']

ratio_adroit = [50]

envs_kitchen = ['kitchen-complete-v0','kitchen-partial-v0', 'kitchen-mixed-v0']
ratio_kitchen = [10, 2]

halfcheetah_replay_envs = ['halfcheetah-medium-replay-v2']
halfcheetah_ratio_replay = [20,4,2]

walker2d_replay_envs = ['walker2d-medium-replay-v2']
walker2d_ratio_replay = [30,6,3]

hopper_replay_envs = ['hopper-medium-replay-v2']
hopper_ratio_replay = [40,8,4]


# maze_env = ['antmaze-umaze-v0']
        # 'hopper-medium-replay-v2',
        # 'halfcheetah-medium-replay-v2',
        # 'walker2d-medium-replay-v2',

        # 'pen-human-v0','pen-cloned-v0','pen-expert-v0', 
        # 'hammer-human-v0','hammer-cloned-v0','hammer-expert-v0', 
        # 'door-human-v0','door-cloned-v0','door-expert-v0', 
        # 'relocate-human-v0','relocate-cloned-v0','relocate-expert-v0'
temp_ratio = [400]

seed = [111]
for env_name in envs_adroit:
    for sd in seed:
        for p in ratio_adroit:
            trajects = []
            dataset = {}
            env = gym.make(env_name)
            ori_dataset = d4rl.qlearning_dataset(env)
            dataset_ = sequence_dataset(env, ori_dataset)
            for n in dataset_:
                trajects.append(n)
            trajects = np.array(trajects)
            whole_ind = np.arange(trajects.shape[0])
            partial_size = trajects.shape[0] // p
            rest = trajects.shape[0] - partial_size
            train_ind, val_ind = train_test_split(whole_ind, train_size=partial_size , test_size=rest, random_state = sd)
            states, actions, next_observations, rewards, terminals  = traj_to_array(trajects[train_ind])
            dataset['observations'] = states
            dataset['actions'] = actions
            dataset['next_observations'] = next_observations
            dataset['rewards'] = rewards
            dataset['terminals'] = terminals
            outs = np.array([dataset])
            path = os.getcwd() + '/' + 'All_samples/' + env_name + '-ratio-' + str(p) + '-seed-' +  str(sd)
            np.save(path,outs)