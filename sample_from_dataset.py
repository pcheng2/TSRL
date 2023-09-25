import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pickle
import collections
class ReplayBuffer(object):
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 quantile, 
                 path_dynamic_ae_network=None,
                 path_bwd_dynamic=None,
                 path_fwd_dynamic=None,
                 max_size=int(1e6), 
                 device='cuda'):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.next_action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_reward = np.zeros((max_size, 1))
        self.nn_reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.state_buffer = []
        self.action_buffer = []
        self.next_state_buffer = []
        self.next_action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []

        
        self.dynamic_ae_network = pickle.load(open(path_dynamic_ae_network, 'rb'))
        self.fwd_dynamic = pickle.load(open(path_fwd_dynamic, 'rb'))
        self.bwd_dynamic = pickle.load(open(path_bwd_dynamic, 'rb'))
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.quantile = quantile
        
        self.device = torch.device(device)

    # 1. Offline RL add data function: add_data_to_buffer -> convert_buffer_to_numpy_dataset -> cat_new_dataset
    def add_data_to_buffer(self, state, action, reward, done):
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)

    # 2. Offline RL add data function: add_data_to_buffer -> convert_buffer_to_numpy_dataset -> cat_new_dataset
    def convert_buffer_to_numpy_dataset(self):
        return np.array(self.state_buffer), \
               np.array(self.action_buffer), \
               np.array(self.reward_buffer), \
               np.array(self.done_buffer)

    # 3. Offline RL add data function: add_data_to_buffer -> convert_buffer_to_numpy_dataset -> cat_new_dataset
    def cat_new_dataset(self, dataset):
        new_state, new_action, new_reward, new_done = self.convert_buffer_to_numpy_dataset()

        state = np.concatenate([dataset['observations'], new_state], axis=0)
        action = np.concatenate([dataset['actions'], new_action], axis=0)
        reward = np.concatenate([dataset['rewards'].reshape(-1, 1), new_reward.reshape(-1, 1)], axis=0)
        done = np.concatenate([dataset['terminals'].reshape(-1, 1), new_done.reshape(-1, 1)], axis=0)

        # free the buffer when you have converted the online sample to offline dataset
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        return {
            'observations': state,
            'actions': action,
            'rewards': reward,
            'terminals': done,
        }

    # TD3 add data function
    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # Offline and Online sample data from replay buffer function
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)  ###################################
        # print('type',type(self.current_z[ind]))
        # print('type',type(self.state[ind]))
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device), 
            
            self.current_z_state[ind],
            self.current_z_action[ind],
            self.next_z_state[ind], 
            
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def sample_lambda(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)  ###################################

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),  ####################################
            torch.FloatTensor(self.next_action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def split_dataset(self, env, dataset, terminate_on_end=False, ratio=10):
        """
            Returns datasets formatted for use by standard Q-learning algorithms,
            with observations, actions, next_observations, rewards, and a terminal
            flag.

            Args:
                env: An OfflineEnv object.
                dataset: An optional dataset to pass in for processing. If None,
                    the dataset will default to env.get_dataset()
                terminate_on_end (bool): Set done=True on the last timestep
                    in a trajectory. Default is False, and will discard the
                    last timestep in each trajectory.
                **kwargs: Arguments to pass to env.get_dataset().
                ratio=N: split the dataset into N peaces

            Returns:
                A dictionary containing keys:
                    observations: An N/ratio x dim_obs array of observations.
                    actions: An N/ratio x dim_action array of actions.
                    next_observations: An N/ratio x dim_obs array of next observations.
                    rewards: An N/ratio-dim float array of rewards.
                    terminals: An N/ratio-dim boolean array of "done" or episode termination flags.
            """
        dataset_ = self.sequence_dataset(env, dataset)
        trajects = []
        for n in dataset_:
            trajects.append(n)
        trajects = np.array(trajects)
        datasize = len(trajects)
        print('datasize', datasize)
        whole_ind = np.arange(datasize)
        partial_data = int(datasize // ratio)
        
        if ratio == 1:
            trajs = trajects[whole_ind]
        else:
            data_ind, _ = train_test_split(whole_ind, train_size=partial_data , test_size=10, random_state = 42)
            trajs = trajects[data_ind]
            
        states, actions, next_states, next_actions, rewards, done_bool = self.traj_to_array(trajs)

        return {
            'observations': states,
            'actions': actions,
            'next_observations': next_states,
            'next_actions': next_actions,
            'rewards': rewards,
            'terminals': done_bool,
        }

    def convert_D4RL(self, dataset, ratio, scale_rewards=False, scale_state=False):
        """
        convert the D4RL dataset into numpy ndarray, you can select whether normalize the rewards and states
        :param dataset: d4rl dataset, usually comes from env.get_dataset or replay_buffer.split_dataset
        :param scale_rewards: whether scale the reward to [0, 1]
        :param scale_state: whether scale the state to standard gaussian distribution ~ N(0, 1)
        :return: the mean and standard deviation of states
        """
        if ratio != 1:
            dataset_size = len(dataset['observations']) -1
            print('dataset_size',dataset_size)
            dataset['terminals'] = dataset['terminals'][:-1]
            dataset['rewards'] = dataset['rewards'][:-1]
            dataset['terminals'] = np.squeeze(dataset['terminals'])
            dataset['rewards'] = np.squeeze(dataset['rewards'])

            nonterminal_steps, = np.where(
                np.logical_and(
                    np.logical_not(dataset['terminals']),
                    np.arange(dataset_size) < dataset_size - 1))

            print('Found %d non-terminal steps out of a total of %d steps.' % (
                len(nonterminal_steps), dataset_size))
            dataset['observations'] = dataset['observations'][:-1]
            dataset['next_observations'] = dataset['next_observations'][:-1]
            action = dataset['actions'][:-1]
            next_action = dataset['actions'][1:]
            
            self.state = dataset['observations'][nonterminal_steps]
            self.action = action[nonterminal_steps]
            self.next_state = dataset['next_observations'][nonterminal_steps]  ####################################
            self.next_action = next_action[nonterminal_steps]
            self.reward = dataset['rewards'][nonterminal_steps].reshape(-1, 1)
            self.not_done = 1. - dataset['terminals'][nonterminal_steps].reshape(-1, 1)
            self.size = self.state.shape[0]
        else:
            
            dataset_size = len(dataset['observations']) + 1
            print('dataset_size',dataset_size)
            dataset['terminals'] = np.squeeze(dataset['terminals'])
            dataset['rewards'] = np.squeeze(dataset['rewards'])

            nonterminal_steps, = np.where(
                np.logical_and(
                    np.logical_not(dataset['terminals']),
                    np.arange(dataset_size) < dataset_size - 1))

            print('Found %d non-terminal steps out of a total of %d steps.' % (
                len(nonterminal_steps), dataset_size))

            self.state = dataset['observations'][nonterminal_steps]
            self.action = dataset['actions'][nonterminal_steps]
            self.next_state = dataset['next_observations'][nonterminal_steps]  ####################################
            self.next_action = dataset['next_actions'][nonterminal_steps]
            self.reward = dataset['rewards'][nonterminal_steps].reshape(-1, 1)
            self.not_done = 1. - dataset['terminals'][nonterminal_steps + 1].reshape(-1, 1)
            self.size = self.state.shape[0]

        # min_max normalization
        if scale_rewards:
            r_max = np.max(self.reward)
            r_min = np.min(self.reward)
            self.reward = (self.reward - r_min) / (r_max - r_min)

        s_mean = self.state.mean(0, keepdims=True)
        s_std = self.state.std(0, keepdims=True)

        # standard normalization
        if scale_state:
            self.state = (self.state - s_mean) / (s_std + 1e-5)
            self.next_state = (self.next_state - s_mean) / (s_std + 1e-5)

        current_z, current_z_state, current_z_act = self.dyna_encoding(self.state,self.action)   
        current_zp, next_zp_state, _ = self.dyna_encoding(self.next_state,self.action)
        
        current_delta_z = self.dyna_pred(current_z)
        current_delta_zp = self.re_dyna_pred(current_zp)
        
        # print('current_delta_z', current_delta_z.shape)
        pred_current_zp_s = current_delta_z + current_z_state
        pred_current_zp = torch.cat((pred_current_zp_s, current_z_act), -1)
        pred_current_delta_zp = self.re_dyna_pred(pred_current_zp)
        pred_current_z_state = pred_current_delta_zp  + next_zp_state
        
        norm_std = 0.01 * torch.std(current_z_state,dim=0, keepdim=True).detach()
        
        z_s_mean = torch.mean(current_z_state,dim=0,keepdim=True)
        z_s_std = torch.std(current_z_state,dim=0, keepdim=True)
        z_std = torch.std(current_z,dim=0, keepdim=True)
        z_mean = torch.mean(current_z,dim=0, keepdim=True)
        
        z_a_std = torch.std(current_z_act, dim=0, keepdim=True)
        
        current_consist_loss = torch.sum((current_z_state - pred_current_z_state)**2, -1) 
        consis_max = torch.max(current_consist_loss)
        consis_min = torch.min(current_consist_loss)
        consis_norm = (current_consist_loss - consis_min) / (consis_max - consis_min)
        thresh = consis_norm.quantile(q=self.quantile).item()
        
        
        
        self.current_z = current_z
        self.current_zp = current_zp
        self.current_z_state = current_z_state
        self.pred_current_z_state = pred_current_z_state
        self.current_z_action = current_z_act
        self.current_delta_z = current_delta_z
        self.current_delta_zp = current_delta_zp
        self.next_z_state = next_zp_state
        
        return s_mean, s_std, norm_std, thresh, consis_max, \
               consis_min, torch.mean(current_consist_loss), \
                z_mean, z_std, \
                z_s_mean, z_s_std, z_a_std


    def normalize_states(self, eps=1e-3):
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std

    def traj_to_array(self, trajs):
        for traj_num in range(len(trajs)):
            if traj_num == 0:
                states = trajs[traj_num]['observations']
                actions = trajs[traj_num]['actions']
                rewards = trajs[traj_num]['rewards']
                terminals = trajs[traj_num]['terminals']
            else:
                states = np.concatenate((states, trajs[traj_num]['observations']), 0)
                actions = np.concatenate((actions, trajs[traj_num]['actions']), 0)
                rewards = np.concatenate((rewards, trajs[traj_num]['rewards']), 0)
                terminals = np.concatenate((terminals, trajs[traj_num]['terminals']), 0)

        # print(states)
        data_size = len(states)
        state = states[:data_size-1]
        next_state = states[1:data_size]

        action = actions[:data_size-1]
        next_action = actions[1:data_size]

        dones = np.array([bool(t) for t in terminals])

        return state, action, next_state, next_action, rewards, dones
    def sequence_dataset(self, env, dataset):
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
        # print(dataset.keys())
        if 'next_observations' not in dataset.keys():
            keys = ['actions', 'observations', 'rewards', 'terminals', 'timeouts']
        else:
            keys = ['actions', 'observations', 'next_observations', 'rewards', 'terminals', 'timeouts']

        N = dataset['rewards'].shape[0]
        data_ = collections.defaultdict(list)

        # The newer version of the dataset adds an explicit
        # timeouts field. Keep old method for backwards compatability.
        use_timeouts = False
        if 'timeouts' in dataset:
            use_timeouts = True

        episode_step = 0
        for i in range(N):
            done_bool = bool(dataset['terminals'][i])
            if use_timeouts:
                final_timestep = dataset['timeouts'][i]
            else:
                final_timestep = (episode_step == env._max_episode_steps - 1)

            for k in keys:
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
    
    def dyna_encoding(self,state,action):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)   
            z_input = torch.cat((state,action),1)
            # print(state.shape) # for pen envs
            # print(action.shape)
            z, state_decode, action_decode = self.dynamic_ae_network(z_input)
            
            z_state = z[:,:self.state_dim]
            # print('self.z_state',z_state.shape)
            z_act = z[:,self.state_dim:]
            # print('self.z_act',z_act.shape)
            return z, z_state, z_act
        
    def dyna_pred(self,z):
        with torch.no_grad():
            pred_z = self.fwd_dynamic(z)
        return pred_z

    def re_dyna_pred(self,z):
        with torch.no_grad():
            pred_z = self.bwd_dynamic(z)
        return pred_z