from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from baselines.common.atari_wrappers import make_atari, wrap_deepmind


def create_env(env_id):
    env = make_atari(env_id)
    return wrap_deepmind(env, episode_life=False, frame_stack=True, clip_rewards=False, scale=True)


def create_random_agent(env_id):
    input_dim, output_dim = get_input_output_dim(env_id)
    return Agent(Policy(input_dim, output_dim))


@lru_cache(maxsize=8)
def get_input_output_dim(env_id):
    env = create_env(env_id)
    input_dim = env.observation_space.shape[-1]
    output_dim = env.action_space.n
    return input_dim, output_dim


def reset_params(module):
    if hasattr(module, 'weight'):
        nn.init.normal_(module.weight)
        w = module.weight
        w = w.view(w.size(0), -1)
        w = w / torch.norm(w, p=2, dim=1, keepdim=True)
        module.weight.data = w.view_as(module.weight)
    if hasattr(module, 'bias'):
        module.bias.zero_()


class Policy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, output_dim)

        for p in self.parameters():
            p.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(reset_params)

    def mutate(self, sigma):
        for p in self.parameters():
            p += sigma * torch.randn_like(p)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Agent(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.reset_stats()

    def reset_stats(self):
        self.ep_rew = 0
        self.ep_len = 0

    def fitness(self):
        return self.ep_rew

    def forward(self, obs, device=None):
        obs = torch.from_numpy(np.array(obs)).transpose(0, 2).unsqueeze(0)
        if device is not None:
            obs = obs.to(device)
        a = self.policy(obs)
        a = torch.argmax(a).item()
        return a

    def mutate(self, sigma):
        self.policy.mutate(sigma)
        self.reset_stats()
