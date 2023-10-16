import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from tdm import train_network
import gym
import d4rl
import wandb
import argparse
 
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
            partial_path = '../utils/small_samples/{}-ratio-{}.npy'.format(args.env_name,int(args.ratio))
            dataset = np.load(partial_path, allow_pickle=True)[0]
            total_states = dataset['observations']
            total_actions = dataset['actions']
        valid_size = 100
        train_size = total_states.shape[0] - valid_size
        
    if args.ratio == 1:
        dataset = env.get_dataset()
        total_states = dataset['observations']
        total_actions = dataset['actions']
        valid_size = 10000
    else:
        partial_path = '../utils/small_samples/{}-ratio-{}.npy'.format(args.env_name,int(args.ratio))
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

    params['loss_weight_state_decoder'] = 1 
    params['loss_weight_act_decoder'] = 1
    params['losss_model_consist'] = 0.1 
    params['loss_weight_dynamic_z_s'] = 0.1 
    params['loss_weight_dynamic_z_s_decode'] = 1
    params['learning_rate'] = 3e-4
    params['l1_rate'] = 1e-5
    
    params['pre_train_epoch'] = args.pre_train_epoch
    params['activation'] = args.activation
    params['max_epochs'] = args.epoch 
    params['data_path'] = 'tdm_models/tdm-Env_{}-ratio_{}/'.format(args.env_name, args.ratio)
    params['device'] = args.device
    params['seed'] = args.seed
    
    return params,training_data,validation_data

def main():
    wandb.init(project="TDM_Models", entity="")
    # Parameters
    parser = argparse.ArgumentParser(description='Train Hopper-v2 with dynamic')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--env_name', help='choose your mujoco env') #  hopper-medium-v2
    parser.add_argument('--epoch', default=500, type=int, help='specific training epoch')
    parser.add_argument('--ratio', default=1, type=float, help='specific data size') # [0, 1]
    parser.add_argument('--batch_size', default=512, type=int, help='specific batch size')
    parser.add_argument('--activation', default='relu', help='specific activation function')
    parser.add_argument('--pre_train_epoch', default=0, type=int, help='specific pre-training epoch')
    parser.add_argument('--seed', default=123, type=int)

    args = parser.parse_args()
    params,training_data,validation_data = pre_processing(args)
    
    wandb.run.name = f"TDM-env_{args.env_name}-ratio_{args.ratio}"
    wandb.config.update(params)
    train_network(training_data, validation_data, params)
    print('--------------------------------finished--------------------------------')

if __name__ == '__main__':
    main()