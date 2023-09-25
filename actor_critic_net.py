import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal, Normal

MEAN_MIN = -9.0
MEAN_MAX = 9.0
LOG_STD_MIN = -5
LOG_STD_MAX = 2
EPS = 1e-7

class Actor(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, device):
        super(Actor, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(num_state, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.mu_head = nn.Linear(num_hidden, num_action)
        self.sigma_head = nn.Linear(num_hidden, num_action)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        log_sigma = torch.clamp(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        action = a_distribution.rsample()

        logp_pi = a_distribution.log_prob(action).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=1)
        logp_pi = torch.unsqueeze(logp_pi, dim=1)

        action = torch.tanh(action)
        return action, logp_pi, a_distribution

    def get_log_density(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        y = torch.clip(y, -1. + EPS, 1. - EPS)
        y = torch.atanh(y)

        mu = torch.clip(mu, MEAN_MIN, MEAN_MAX)
        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        logp_pi = a_distribution.log_prob(y).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - y - F.softplus(-2 * y))).sum(axis=1)
        logp_pi = torch.unsqueeze(logp_pi, dim=1)
        return logp_pi

    def get_action(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        action = a_distribution.rsample()
        action = torch.tanh(action)
        return action

class Actor_deterministic(nn.Module):
    def __init__(self, num_state, num_action, num_hidden,dropout_rate, device):
        super(Actor_deterministic, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(num_state, num_hidden)
        self.drop = torch.nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.action = nn.Linear(num_hidden, num_action)
        
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        a = F.relu(self.fc1(x))
        a = self.drop(a)
        a = F.relu(self.fc2(a))
        a = self.drop(a)
        a = self.action(a)
        return torch.tanh(a)
    
# class Actor_deterministic(nn.Module):
#     def __init__(self, num_state, num_action, num_hidden,dropout_rate, device):
#         super(Actor_deterministic, self).__init__()
#         self.device = device
#         self.fc1 = nn.Linear(num_state, num_hidden)
#         self.drop_1 = torch.nn.Dropout(p=dropout_rate)
#         self.fc2 = nn.Linear(num_hidden, num_hidden)
#         self.action = nn.Linear(num_hidden, num_action)
        
#     def forward(self, x):
#         if isinstance(x, np.ndarray):
#             x = torch.tensor(x, dtype=torch.float).to(self.device)
#         a = F.relu(self.fc1(x))
#         a = self.drop_1(a)
#         a = F.relu(self.fc2(a))
#         a = self.drop_1(a)
#         a = self.action(a)
#         return torch.tanh(a)


# Double Q_net
class Double_Critic(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, dropout_rate, device):
        super(Double_Critic, self).__init__()
        self.device = device
        self.drop_layer = torch.nn.Dropout(p=dropout_rate)
        # Q1 architecture
        self.fc1 = nn.Linear(num_state+num_action, num_hidden)
        self.ln1 = nn.LayerNorm((num_hidden,))
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.ln2 = nn.LayerNorm((num_hidden,))
        self.fc3 = nn.Linear(num_hidden, 1)
        
        # Q2 architecture
        self.fc4 = nn.Linear(num_state+num_action, num_hidden)
        self.ln4 = nn.LayerNorm((num_hidden,))
        self.fc5 = nn.Linear(num_hidden, num_hidden)
        self.ln5 = nn.LayerNorm((num_hidden,))
        self.fc6 = nn.Linear(num_hidden, 1)


    def forward(self, x, y):
        sa = torch.cat([x, y], 1)
        q1 = self.fc1(sa)
        # q1 = self.ln1(q1)
        q1 = F.relu(q1)
        q1 = self.fc2(q1)
        # q1 = self.ln2(q1)
        q1 = F.relu(q1)
        q1 = self.fc3(q1)
        
        q2 = self.fc4(sa)
        # q2 = self.ln4(q2)
        q2 = F.relu(q2)
        q2 = self.fc5(q2)
        # q2 = self.ln5(q2)
        q2 = F.relu(q2)
        q2 = self.fc6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.fc1(sa)
        # q1 = self.ln1(q1)
        q1 = F.relu(q1)
        # q1 = self.drop_layer(q1)
        q1 = self.fc2(q1)
        # q1 = self.ln2(q1)
        q1 = F.relu(q1)
        # q1 = self.drop_layer(q1)
        q1 = self.fc3(q1)
        return q1