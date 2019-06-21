import argparse

import torch
import numpy as np
from tqdm import trange

from atari import create_env, create_random_agent
from utils import set_seeds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='FrostbiteNoFrameskip-v0')
    parser.add_argument('--agent', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gif', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seeds(args.seed)

    env = create_env(args.env)
    env.seed(args.seed)
    print("Loading environment", args.env)
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    agent = create_random_agent(args.env)
    agent.load_state_dict(torch.load(args.agent))

    rewards = []
    for _ in trange(30):
        ep_rew = 0
        obs = env.reset()
        done = False
        while not done:
            act = agent(obs)
            obs, rew, done, _ = env.step(act)
            ep_rew += rew
        rewards.append(ep_rew)

    print(f"ep_rew = {np.mean(rewards):.2f} +- {np.std(rewards):.2f}")


if __name__ == '__main__':
    main()
