import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal, Normal

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

# Double Q_net
class Double_Critic(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, dropout_rate, device):
        super(Double_Critic, self).__init__()
        self.device = device
        self.drop_layer = torch.nn.Dropout(p=dropout_rate)
        # Q1 architecture
        self.fc1 = nn.Linear(num_state+num_action, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, 1)
        
        # Q2 architecture
        self.fc4 = nn.Linear(num_state+num_action, num_hidden)
        self.fc5 = nn.Linear(num_hidden, num_hidden)
        self.fc6 = nn.Linear(num_hidden, 1)


    def forward(self, x, y):
        sa = torch.cat([x, y], 1)
        q1 = self.fc1(sa)
        q1 = F.relu(q1)
        q1 = self.fc2(q1)
        q1 = F.relu(q1)
        q1 = self.fc3(q1)
        
        q2 = self.fc4(sa)
        q2 = F.relu(q2)
        q2 = self.fc5(q2)
        q2 = F.relu(q2)
        q2 = self.fc6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.fc1(sa)
        q1 = F.relu(q1)
        q1 = self.fc2(q1)
        q1 = F.relu(q1)
        q1 = self.fc3(q1)
        return q1