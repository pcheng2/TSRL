import numpy as np
import d4rl
import gym
from sklearn.model_selection import train_test_split
import collections
import os 
import argparse

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

def create_samples(env_name, ratio):
    '''
    split data by trajectories
    '''
    trajects = []
    dataset = {}
    env = gym.make(env_name)
    ori_dataset = d4rl.qlearning_dataset(env)
    dataset_ = sequence_dataset(env, ori_dataset)
    for n in dataset_:
        trajects.append(n)
    trajects = np.array(trajects)
    whole_ind = np.arange(trajects.shape[0])
    partial_size = int(trajects.shape[0] // ratio)
    rest = trajects.shape[0] - partial_size
    train_ind, _ = train_test_split(whole_ind, train_size=partial_size , test_size=rest, random_state = 111)
    states, actions, next_observations, rewards, terminals  = traj_to_array(trajects[train_ind])
    dataset['observations'] = states
    dataset['actions'] = actions
    dataset['next_observations'] = next_observations
    dataset['rewards'] = rewards
    dataset['terminals'] = terminals
    outs = np.array([dataset])
    path = os.getcwd() + '/' + 'small_samples/' + env_name + '-ratio-' + str(ratio)
    np.save(path,outs)
            
def main():
    parser = argparse.ArgumentParser(description='Train TDM')
    parser.add_argument('--env_name', default='hopper-medium-v2', help='choose your mujoco env')
    parser.add_argument('--ratio', default=100, type=float, help='data ratio')
    args = parser.parse_args()
    create_samples(args.env_name, args.ratio)

if __name__ == '__main__':
  main()
  