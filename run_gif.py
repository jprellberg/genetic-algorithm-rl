import argparse

import torch
from PIL import Image

from atari import create_env, create_random_agent
from utils import set_seeds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='FrostbiteNoFrameskip-v0')
    parser.add_argument('--agent', type=str, required=True)
    parser.add_argument('--gif', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
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

    frames = []

    obs = env.reset()
    done = False
    while not done:
        rgb = env.render('rgb_array')
        act = agent(obs)
        obs, rew, done, _ = env.step(act)
        img = Image.fromarray(rgb)
        frames.append(img)

    frames[0].save(args.gif, format='GIF', append_images=frames[1:], save_all=True, duration=50, loop=0, optimize=False)


if __name__ == '__main__':
    main()
