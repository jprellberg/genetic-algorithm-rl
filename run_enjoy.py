import argparse
from time import sleep

import torch

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

    while True:
        ep_rew = 0
        ep_len = 0
        obs = env.reset()
        done = False
        while not done:
            env.render()
            sleep(0.03)
            act = agent(obs)
            obs, rew, done, _ = env.step(act)
            ep_rew += rew
            ep_len += 1
        print(f"ep_rew={ep_rew:.2f} ep_len={ep_len}")


if __name__ == '__main__':
    main()
