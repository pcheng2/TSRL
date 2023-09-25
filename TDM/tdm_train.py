import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from tdm import train_network
import gym
import d4rl
import wandb
import argparse
import collections

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

    keys = ['actions', 'observations', 'rewards', 'terminals', 'timeouts']
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

def traj_to_array(trajs):
    for traj_num in range(len(trajs)):
        if traj_num == 0:
            states = trajs[traj_num]['observations']
            actions = trajs[traj_num]['actions']
        else:
            states = np.concatenate((states, trajs[traj_num]['observations']), 0)
            actions = np.concatenate((actions, trajs[traj_num]['actions']), 0)

    return states, actions
    
def pre_processing(args):
    params = {}
    training_data = {}
    validation_data = {}
    params['env_name'] = args.env_name
    env = gym.make(args.env_name)
    params['act_dim'] = env.action_space.shape[0]
    params['state_dim'] = env.observation_space.shape[0]
    
    if 'pen' or 'relocate' or 'door' or 'hammer' in args.env_name:
        if args.ratio == 1:
            dataset = env.get_dataset()
            total_states = dataset['observations']
            total_actions = dataset['actions']
        else:
            partial_path = '/data/pcheng/sindy/SINDy_RL/All_samples/{}-ratio-{}-seed-111.npy'.format(args.env_name,int(args.ratio))
            dataset = np.load(partial_path, allow_pickle=True)[0]
            total_states = dataset['observations']
            total_actions = dataset['actions']
        valid_size = 100
        train_size = total_states.shape[0] - valid_size
        
    if "kitchen" in args.env_name:
        dataset = env.get_dataset()
        total_states = dataset['observations']
        total_actions = dataset['actions']
        valid_size = 10
        train_size = total_states.shape[0] - valid_size
        
    if args.ratio == 1:
        if args.low_speed:
            partial_path = '/data/pcheng/sindy/SINDy_RL/low_speed_data/{}-TRUE_low_speed-quantile_{}.npy'.format(args.env_name,args.speed_thresh)
            dataset = np.load(partial_path, allow_pickle=True)[0]
            total_states = dataset['observations']
            total_actions = dataset['actions']
            valid_size = 1000
        else:
            dataset = env.get_dataset()
            total_states = dataset['observations']
            total_actions = dataset['actions']
            valid_size = 10000
    else:
        partial_path = '/data/pcheng/sindy/SINDy_RL/All_samples/{}-ratio-{}-seed-111.npy'.format(args.env_name,int(args.ratio))
        dataset = np.load(partial_path, allow_pickle=True)[0]
        total_states = dataset['observations']
        total_actions = dataset['actions']
        valid_size = 1000
        
    params['latent_s_dim'] = env.observation_space.shape[0] 
    params['latent_act_dim'] = env.action_space.shape[0]
    params['latent_out'] = params['latent_s_dim'] + params['latent_act_dim']  
    params['input_dim'] = params['state_dim'] + params['act_dim']
    
    train_size = total_states.shape[0] - valid_size
    train_state = total_states[:train_size]
    train_action = total_actions[:train_size]
    valid_state = total_states[train_size:]
    valid_action = total_actions[train_size:]

    s_mean = total_states.mean(0, keepdims=True)
    s_std = total_states.std(0, keepdims=True)

    params['batch_size'] = args.batch_size
    params['train_size'] = train_size
    params['batch_num'] = train_size // params['batch_size']

    ### construct training dataset
    
    training_data['s'] = train_state[:-1]
    training_data['sp'] = train_state[1:] 
    training_data['act'] = train_action[:-1]
    training_data['next_act'] = train_action[1:]
    
    training_data['s'] = (training_data['s'] - s_mean) / (s_std + 1e-5)
    training_data['sp'] = (training_data['sp'] - s_mean) / (s_std + 1e-5)
    training_data['ds'] = training_data['sp'] - training_data['s']
    training_data['dsp'] = -training_data['ds']

    trainning_num_data, trainning_act_dim = training_data['act'].shape
    training_data['da'] = np.zeros([trainning_num_data, trainning_act_dim], dtype=float)
    
    validation_data['s'] = valid_state[:-1]
    validation_data['sp'] = valid_state[1:]
    validation_data['act'] = valid_action[:-1]
    validation_data['next_act'] = valid_action[1:]
    
    validation_data['s'] = (validation_data['s'] - s_mean) / (s_std + 1e-5)
    validation_data['sp'] = (validation_data['sp'] - s_mean) / (s_std + 1e-5)
    validation_data['ds'] = validation_data['sp'] - validation_data['s']
    validation_data['dsp'] = -validation_data['ds']
    
    valid_num_data, valid_act_dim = validation_data['act'].shape
    validation_data['da'] = np.zeros([valid_num_data, valid_act_dim], dtype=float)

    #cheetah-expert

    params['loss_weight_state_decoder'] = 1 # 1/0.5 weighting of the autoencoder reconstruction in the loss function
    params['loss_weight_act_decoder'] = 1
    params['losss_model_consist'] = 0.1 #1
    params['loss_weight_dynamic_z_s'] = 0.1 #weighting of the Dynamic prediction in the latent space in the loss function
    params['loss_weight_dynamic_z_s_decode'] = 1 # 1
    params['pre_train_epoch'] = args.pre_train_epoch
    params['activation'] = args.activation #activation function to be used in the network

    params['learning_rate'] = 3e-4 #learning rate passed to the adroit 1e-3 , locomotion 3e-4
    params['l1_rate'] = 1e-5
    params['wd_rate'] = 1e-5
    
    params['max_epochs'] = args.epoch #how many epochs to run the training procedure for
    if args.low_speed:
        params['data_path'] = 'tdm_models/Low-speed-Env_{}-ratio_{}/'.format(args.env_name, args.ratio)
    else:
        params['data_path'] = 'tdm_models/WD-Env_{}-ratio_{}/'.format(args.env_name, args.ratio)

    params['device'] = args.device
    params['seed'] = args.seed
    return params,training_data,validation_data

def main():
    
    # Parameters
    parser = argparse.ArgumentParser(description='Train Hopper-v2 with dynamic')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--env_name', help='choose your mujoco env') #  hopper-medium-v2
    parser.add_argument('--epoch', default=500, type=int, help='specific epoch times')
    parser.add_argument('--ratio', default=1, type=float, help='specific train_size') # [0, 1]
    parser.add_argument('--speed_thresh', default=0, type=float, help='specific train_size') # [0, 1]
    parser.add_argument('--batch_size', default=512, type=int, help='specific batch_size')
    parser.add_argument('--activation', default='relu', help='specific activation function')
    parser.add_argument('--low_speed', default=False, type=bool)
    parser.add_argument('--pre_train_epoch', default=0, type=int, help='specific activation function')
    parser.add_argument('--seed', default=123, type=int, help='specific activation function')

    args = parser.parse_args()
    wandb.init(project="TDM_Models", entity="pcheng")
    # setup mujoco environment and SBAC agent
    env_name = args.env_name
    print('-----------------{}-----------------'.format(env_name))
    params,training_data,validation_data = pre_processing(args)
    wandb.run.name = f"TDM-WD-Pretrain_{args.pre_train_epoch}-NN_512-{env_name}-seed_{args.seed}-epoch_{args.epoch}-low_speed_{args.low_speed}_{args.speed_thresh}-batch_size_{args.batch_size}-encoder_dim_{params['latent_out']}-ratio_{args.ratio}-BaiDu"
    wandb.config.update(params)
    train_network(training_data, validation_data, params)
    print('--------------------------------finished--------------------------------')

if __name__ == '__main__':
    main()

# export WANDB_API_KEY=ea1eb211c31bd14320ec5da1ed751ff43eaed121

# hopper-random-v2      hopper-medium-v2        hopper-medium-replay-v2        hopper-medium-expert-v2       hopper-expert-v2 
# halfcheetah-random-v2 halfcheetah-medium-v2   halfcheetah-medium-replay-v2   halfcheetah-medium-expert-v2  halfcheetah-expert-v2 
# walker2d-random-v2    walker2d-medium-v2      walker2d-medium-replay-v2      walker2d-medium-expert-v2     walker2d-expert-v2 
# pen-human-v1          pen-cloned-v1           pen-expert-v1
# hammer-human-v1       hammer-cloned-v1        hammer-expert-v1 
# door-human-v1         door-cloned-v1          door-expert-v1 
# relocate-human-v1     relocate-cloned-v1      relocate-expert-v1

# antmaze-umaze-v0    antmaze-umaze-diverse-v0  antmaze-medium-diverse-v0  antmaze-medium-play-v0   antmaze-large-diverse-v0   antmaze-large-play-v0

# export CUDA_VISIBLE_DEVICES=2


# LOCOMOTION                    ori         10k     50k     100k
# expert/medium/random:         1M          100	    20	    10	
# medium-expert:                2M          200	    40	    20	
# halfcheetah-medium-replay:    200k        20	    4	    2
# walker2d-medium-replay:       300k        30	    6	    3
# hopper-medium-replay:         400k        40	    8	    4
# -----------------------------------------------------
# Adroit                        ori         10k    5k
# human-v1:                     5k                
# cloned-v1:                    500k        50     100
# expert-v1:                    500k        50     100   
# kitchen-complete-v0:          3680 
# kitchen-partial-v0:           130k        10     2
# kitchen-mixed-v0:             130k        10     2

# -------------------------------------------------------------
# antmaze-umaze-v0:             1M          100    
# antmaze-umaze-diverse-v0:     1M          100    
# antmaze-medium-diverse-v0:    1M          100    
# antmaze-medium-play-v0:       1M          100    
# antmaze-large-diverse-v0:     1M          100    
# antmaze-large-play-v0:        1M          100    



# nohup python tdm_train.py --env_name halfcheetah-medium-v2 --ratio 1 --pre_train_epoch 20 --epoch 200  > logs/dynamic-halfcheetah-medium-v2-1.log 2>&1 &

# nohup python tdm_train.py --env_name halfcheetah-medium-expert-v2 --ratio 200 --pre_train_epoch 200 --epoch 2000  > logs/dynamic-halfcheetah-medium-v2-100.log 2>&1 &

# nohup python tdm_train.py --env_name halfcheetah-medium-replay-v2 --ratio 20 --pre_train_epoch 200 --epoch 2000  > logs/dynamic-halfcheetah-medium-replay-v2-1.log 2>&1 &

# nohup python tdm_train.py --env_name halfcheetah-expert-v2 --ratio 1 --pre_train_epoch 20 --epoch 200  > logs/dynamic-halfcheetah-expert-v2-1.log 2>&1 &


# nohup python tdm_train.py --env_name walker2d-medium-v2 --ratio 1 --pre_train_epoch 20 --epoch 200  > logs/dynamic-walker2d-medium-v2-1.log 2>&1 &

# nohup python tdm_train.py --env_name walker2d-medium-replay-v2 --ratio 1 --pre_train_epoch 20 --epoch 200  > logs/dynamic-walker2d-medium-replay-v2-1.log 2>&1 &

# nohup python tdm_train.py --env_name walker2d-medium-expert-v2 --ratio 1 --pre_train_epoch 20 --epoch 200  > logs/dynamic-walker2d-medium-expert-v2-20.log 2>&1 &

# nohup python tdm_train.py --env_name walker2d-expert-v2 --ratio 1 --pre_train_epoch 20 --epoch 200  > logs/dynamic-walker2d-expert-v2-1.log 2>&1 &


# nohup python tdm_train.py --env_name hopper-medium-v2 --ratio 1 --pre_train_epoch 20 --epoch 200  > logs/dynamic-walker2d-random-v2-10.log 2>&1 &

# nohup python tdm_train.py --env_name hopper-medium-expert-v2 --ratio 200 --pre_train_epoch 200 --epoch 2000  > logs/dynamic-walker2d-random-v2-10.log 2>&1 &

# nohup python tdm_train.py --env_name hopper-medium-replay-v2 --ratio 1 --pre_train_epoch 20 --epoch 200  > logs/dynamic-hopper-medium-replay-v2-40.log 2>&1 &

# nohup python tdm_train.py --env_name hopper-expert-v2 --ratio 1 --pre_train_epoch 20 --epoch 200 > logs/dynamic-hopper-expert-v2-100.log 2>&1 &


# nohup python tdm_train.py --env_name kitchen-mixed-v0  --ratio 10  --pre_train_epoch 200  --concat_num 1 --concat True --epoch 2000 > logs/dynamic-kitchen-complete-v0.log 2>&1 &

# nohup python tdm_train.py --env_name relocate-expert-v1 --ratio 50 --pre_train_epoch 0 --epoch 200000 > logs/dynamic-relocate-cloned-v1.log 2>&1 &
