from functools import partial

import numpy as np
import torch
import torch.nn as nn
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

from utils import Flatten


def create_env(env_id):
    env = make_atari(env_id)
    return wrap_deepmind(env, episode_life=False, frame_stack=True, clip_rewards=False, scale=True)


def get_input_output_dim(env):
    input_dim = env.observation_space.shape[-1]
    output_dim = env.action_space.n
    return input_dim, output_dim


class Policy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, obs, device=None):
        obs = torch.from_numpy(np.array(obs)).transpose(0, 2).unsqueeze(0)
        if device is not None:
            obs = obs.to(device)
        a = self.layers(obs)
        return a


def param_init(module, generator):
    if hasattr(module, 'weight'):
        w = module.weight
        w.normal_(0, 1, generator=generator)
        w = w.view(w.size(0), -1)
        w = w / torch.norm(w, p=2, dim=1, keepdim=True)
        module.weight.data = w.view_as(module.weight)
    if hasattr(module, 'bias'):
        module.bias.zero_()


class Agent:
    def __init__(self):
        self.mutations = []
        self.mutate(None)

    def reset_stats(self):
        self.ep_rew = []
        self.ep_len = []

    def num_evals(self):
        return len(self.ep_rew)

    def fitness(self):
        return np.mean(self.ep_rew)

    def get_policy(self, input_dim, output_dim):
        policy = Policy(input_dim, output_dim)
        # Set initial weights
        seed, _ = self.mutations[0]
        gen = new_generator(seed)
        policy.apply(partial(param_init, generator=gen))
        # Apply mutations
        for seed, sigma in self.mutations[1:]:
            gen = new_generator(seed)
            for p in policy.parameters():
                noise = torch.empty_like(p)
                noise.normal_(0, 1, generator=gen)
                p += sigma * noise
        return policy

    def mutate(self, sigma):
        self.mutations.append((new_seed(), sigma))
        self.reset_stats()


def new_seed():
    return torch.randint(torch.iinfo(torch.int64).max, (1,)).item()


def new_generator(seed):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen
