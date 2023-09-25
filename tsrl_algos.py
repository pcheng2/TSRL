import sys
import os
sys.path.append(os.getcwd())
import copy
import torch
import torch.nn as nn
import wandb
import gym
import os
import numpy as np
import d4rl
from torch.distributions import Normal
from Prepare_env import prepare_env
from actor_critic_net import Actor_deterministic, Double_Critic
from sample_from_dataset import ReplayBuffer
import pickle

class TSRL:
    def __init__(self,
                 env_name,
                 num_hidden=512,
                 gamma=0.999,
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_freq=2,
                 alpha=2.5,
                 ratio=1,
                 seed=0,
                 lr_actor=3e-4,
                 lr_critic=3e-4,
                 quantile=1,
                 z_act_weight = 0,
                 inconsis_weight = 0,
                 batch_size=256,
                 Drop_prob=0.5,
                 eval_iter=5,
                 low_speed=False,
                 store_prams=False,
                 augment=False,
                 device='cpu'):
        """
        Facebear's implementation of TD3_BC (A Minimalist Approach to Offline Reinforcement Learning)
        Paper: https://arxiv.org/pdf/2106.06860.pdf
        :param env_name: your gym environment name
        :param num_hidden: the number of the units of the hidden layer of your network
        :param gamma: discounting factor of the cumulated reward
        :param tau: soft update
        :param policy_noise:
        :param noise_clip:
        :param policy_freq: delayed policy update frequency
        :param alpha: the hyper-parameter in equation
        :param ratio:
        :param device:
        """
        super(TSRL, self).__init__()
        # prepare the environment
        self.env_name = env_name
        self.env = gym.make(env_name)
        num_state = self.env.observation_space.shape[0]
        num_action = self.env.action_space.shape[0]
        
        self.augment = augment
        self.quantile = quantile
        self.low_speed = low_speed
        
        self.z_act_weight = z_act_weight
        self.inconsis_weight = inconsis_weight
        self.store_prams = store_prams
        
        path_dynamic_ae_network = 'TDM/tdm_models/Env_{}-ratio_{}/AE_params.pkl'.format(env_name, ratio)
        path_bwd_dynamic = 'TDM/tdm_models/Env_{}-ratio_{}/Dyna_bwd_params.pkl'.format(env_name, ratio)
        path_fwd_dynamic = 'TDM/tdm_models/Env_{}-ratio_{}/Dyna_fwd_params.pkl'.format(env_name, ratio)
        self.device = device
        
        # hyper-parameters
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.policy_noise = policy_noise
        self.policy_freq = policy_freq
        self.evaluate_freq = 3000
        self.noise_clip = noise_clip
        self.alpha = alpha
        self.batch_size = batch_size
        
        self.max_action = 1.
        self.total_it = 0
        self.actor_aug_count = 0
        self.eval_iter = eval_iter
        self.Drop_prob = Drop_prob
        self.seed = seed
        # set seed
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        torch.manual_seed(seed)
        
        self.latent_state_dim = self.env.observation_space.shape[0] #15 8
        self.latent_action_dim = self.env.action_space.shape[0] #5 
        
        self.replay_buffer = ReplayBuffer(num_state, 
                                        num_action, 
                                        quantile,
                                        path_dynamic_ae_network=path_dynamic_ae_network,
                                        path_bwd_dynamic = path_bwd_dynamic,
                                        path_fwd_dynamic = path_fwd_dynamic,
                                        device=self.device)
        
        if ratio != 1:
            partial_path = 'small_samples/{}-ratio-{}-seed-111.npy'.format(self.env_name,int(ratio))
            self.dataset = np.load(partial_path, allow_pickle=True)[0]
        else:
            dataset = self.env.get_dataset()
            # dataset = np.load('/root/sindy/SINDy_RL/low_speed_data/walker2d-medium-v2-low_speed.npy', allow_pickle=True)[0]
            self.dataset = self.replay_buffer.split_dataset(self.env, dataset, ratio=ratio)
            
        self.s_mean, self.s_std, self.norm_std, self.thresh, self.consis_max, self.consis_min, self.consis_mean, self.z_mean, self.z_std, self.z_s_mean, self.z_s_std, self.z_a_std = self.replay_buffer.convert_D4RL(self.dataset, ratio, scale_rewards=False, scale_state=True)
        self.dynamic_ae_network = pickle.load(open(path_dynamic_ae_network, 'rb'))
        self.fwd_dynamic = pickle.load(open(path_fwd_dynamic, 'rb'))
        self.bwd_dynamic = pickle.load(open(path_bwd_dynamic, 'rb'))
        
        # prepare the actor and critic
        self.actor_net = Actor_deterministic(num_state, num_action, num_hidden, self.Drop_prob, device).float().to(device)
        self.actor_target = copy.deepcopy(self.actor_net)
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr=3e-4)

        self.critic_net = Double_Critic(self.latent_state_dim, self.latent_action_dim, num_hidden, self.Drop_prob, device).float().to(device)
        self.critic_target = copy.deepcopy(self.critic_net)
        self.critic_optim = torch.optim.Adam(self.critic_net.parameters(), lr=3e-4)

        # Q and Critic file location
        if self.store_prams:
            self.file_loc = prepare_env(self.alpha, env_name, self.z_act_weight, self.inconsis_weight, ratio)
    
    def tsrl_learn(self, total_time_step=1e+5):
        records = {}
        records['states'] = []
        records['qpos'] = []
        while self.total_it <= total_time_step:
            self.total_it += 1
            # sample data
            state, action, next_state, current_z_state, current_z_action, next_z_state, reward, not_done = self.replay_buffer.sample(self.batch_size)
            # update Critic
            critic_loss_pi = self.train_Q_pi(action, 
                                            current_z_state, 
                                            current_z_action, 
                                            next_state, 
                                            reward, 
                                            not_done)

            # delayed policy update
            if self.total_it % self.policy_freq == 0:
                actor_loss, latent_bc_loss, inconsis_loss, aug_consist_loss, Q_pi_mean = self.train_actor(state, 
                                                                                                          next_state, 
                                                                                                          current_z_action, 
                                                                                                          next_z_state)
                if self.total_it % self.evaluate_freq == 0:
                    evaluate_reward = self.rollout_evaluate()
                    wandb.log({"actor_loss": actor_loss,
                               "latent_bc_loss": latent_bc_loss,
                               "inconsis_loss":inconsis_loss,
                               "Q_pi_loss": critic_loss_pi,
                               "Q_pi_mean": Q_pi_mean,
                               "aug_consist_loss": aug_consist_loss,
                               "evaluate_rewards": evaluate_reward,
                               "aug_thresh": self.thresh,
                               "actor_aug_count": self.actor_aug_count,
                               "it_steps": self.total_it
                               })
                    
            # if self.store_prams:
            #     if self.total_it % 900000 == 0:
            #         self.save_parameters()

    def train_Q_pi(self, 
                   action, 
                   current_z_state, 
                   current_z_action, 
                   next_state, 
                   reward, 
                   not_done):
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            pi_next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            _, pi_next_z_obs, pi_next_z_action = self.dyna_encoding(self.dynamic_ae_network, next_state, pi_next_action)
            
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(pi_next_z_obs, pi_next_z_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.gamma * target_Q

        Q1, Q2 = self.critic_net(current_z_state, current_z_action)
        # Critic loss
        critic_loss = nn.MSELoss()(Q1, target_Q) + nn.MSELoss()(Q2, target_Q)
        
        # Optimize Critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        return critic_loss.cpu().detach().numpy().item()
    
    def train_actor(self, state, next_state, current_z_action, next_z_s):
        action_pi = self.actor_net(state)
        next_action_pi = self.actor_net(next_state)
        pi_z, pi_z_s, pi_z_action= self.dyna_encoding(self.dynamic_ae_network, state, action_pi)
        
        # z, pi_next_z_obs, _= self.dyna_encoding(self.dynamic_ae_network, next_state, action_pi)
        z, pi_next_z_obs, _= self.dyna_encoding(self.dynamic_ae_network, next_state, next_action_pi)
        
        pi_delta_z_s = self.fwd_dynamic(pi_z)
        pred_next_z_s = pi_z_s + pi_delta_z_s
        pred_zp = torch.cat((pred_next_z_s, pi_z_action), -1)
        pi_delta_sp = self.bwd_dynamic(pred_zp)
        pred_z_s = pi_delta_sp + pi_next_z_obs

        inconsis_loss = nn.MSELoss()(pi_z_s, pred_z_s)
        latent_bc_loss = nn.MSELoss()(pi_z_action, current_z_action)
        
        if self.augment:
            self.epsilon = Normal(self.z_s_mean, self.norm_std).sample().detach()
            aug_z_state = pi_z_s + self.epsilon
            aug_next_z_state = next_z_s + self.epsilon
            
            aug_z = torch.cat((aug_z_state, pi_z_action), -1)
            pred_aug_delta_z_s = self.fwd_dynamic(aug_z)
            epsilon_z_s = pred_aug_delta_z_s - pi_delta_z_s
            
            pred_aug_z_next_state = aug_next_z_state + epsilon_z_s 
            aug_zp = torch.cat((pred_aug_z_next_state, pi_z_action), -1)
            
            pred_aug_delta_zp = self.bwd_dynamic(aug_zp)
            pred_aug_z_state = pred_aug_delta_zp + aug_next_z_state
            aug_consist_loss = nn.MSELoss()(aug_z_state, pred_aug_z_state).detach()
            
            if aug_consist_loss <= self.thresh:
                self.actor_aug_count += 1
                pi_z_s = torch.cat((pi_z_s, aug_z_state), 0)
                pi_z_action = torch.cat((pi_z_action, pi_z_action), 0)
        
        # if self.norm:
        #     pi_z_s = (pi_z_s - self.z_s_mean) / (self.z_s_std + 1e-5)
        
        Q1, Q2 = self.critic_net(pi_z_s, pi_z_action)
        Q_pi = torch.min(Q1, Q2)
        # Q_pi = self.critic_net.Q1(pi_z_s, pi_z_action)
        lmbda = self.alpha / Q_pi.abs().mean().detach()
        actor_loss = -lmbda * Q_pi.mean() + self.z_act_weight * latent_bc_loss + self.inconsis_weight * inconsis_loss
        
        # Optimize Actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # update the frozen target models
        for param, target_param in zip(self.critic_net.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor_net.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.cpu().detach().numpy().item(), \
               latent_bc_loss.cpu().detach().numpy().item(), \
               inconsis_loss.cpu().detach().numpy().item(), \
               aug_consist_loss.cpu().detach().numpy().item(), \
               Q_pi.mean().cpu().detach().numpy().item()
        
        
    def rollout_evaluate(self):
        """
        policy evaluation function
        :return: the evaluation result
        """
        ep_rews = []
        for _ in range(self.eval_iter):
            scores = 0
            state = self.env.reset()
            while True:
                state = (state - self.s_mean) / (self.s_std + 1e-5)
                action = self.actor_net(state).cpu().detach().numpy()
                # print(action)
                state, reward, done, _ = self.env.step(action[0])
                # states.append(state)
                # qps.append(self.env.data.qpos[0])
                scores += reward
                if done:
                    break
            scores = d4rl.get_normalized_score(env_name=self.env_name, score=scores) * 100
            ep_rews.append(scores)
        ep_rewards_mean = np.mean(ep_rews)
        return ep_rewards_mean
    
    def save_parameters(self):
        torch.save(self.critic_net.state_dict(), self.file_loc[2])
        torch.save(self.actor_net.state_dict(), self.file_loc[3])

    def load_parameters(self):
        self.critic_net.load_state_dict(torch.load(self.file_loc[2]))
        self.actor_net.load_state_dict(torch.load(self.file_loc[3]))
        
    def dyna_encoding(self,network,state,action):
        z_input = torch.cat((state,action),1)
        z, _, _ = network(z_input)
        z_state = z[:,:self.latent_state_dim]
        z_act = z[:,self.latent_state_dim:]
        return z, z_state, z_act