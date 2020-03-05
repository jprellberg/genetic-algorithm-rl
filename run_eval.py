import argparse
from itertools import count
from pathlib import Path
from time import sleep

import numpy as np
import torch
from PIL import Image

from atari import create_env, get_input_output_dim
from utils import set_seeds, pickle_load


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True)
    parser.add_argument('--history-file', type=str, required=True)
    parser.add_argument('--seed', type=int, default=666666)
    agent = parser.add_mutually_exclusive_group(required=True)
    agent.add_argument('--best-agent', action='store_true')
    agent.add_argument('--best-agent-ensemble', action='store_true')
    agent.add_argument('--last-agent', action='store_true')
    agent.add_argument('--last-agent-ensemble', action='store_true')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--evaluate', type=int)
    mode.add_argument('--render', action='store_true')
    mode.add_argument('--gif', type=str)
    args = parser.parse_args()
    return args


def run_episode(env, policy, step_hook=None):
    ep_rew = 0
    ep_len = 0
    done = False
    obs = env.reset()
    while not done:
        if step_hook:
            step_hook(env)
        act = policy(obs).argmax().item()
        obs, rew, done, _ = env.step(act)
        ep_rew += rew
        ep_len += 1
    return ep_rew, ep_len


def render_human(env):
    env.render()
    sleep(0.03)


class RenderGif:
    def __init__(self):
        self.frames = []

    def __call__(self, env):
        rgb = env.render('rgb_array')
        img = Image.fromarray(rgb)
        self.frames.append(img)

    def save(self, file):
        self.frames[0].save(file, format='GIF', append_images=self.frames[1:],
                            save_all=True, duration=50, loop=0, optimize=False)


class EnsemblePolicy:
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, obs):
        actprob = [p(obs) for p in self.policies]
        return torch.stack(actprob).mean(0)


def get_best_agents_dedup(history, count):
    fit_elite_mean = np.array(history['fit_elite_mean'])
    best = np.argsort(fit_elite_mean)
    agents = [history['agents'][idx][0] for idx in reversed(best)]
    agents_dedup = []
    for i in range(len(agents)):
        if tuple(agents[i].mutations) not in set(tuple(a.mutations) for a in agents_dedup):
            agents_dedup.append(agents[i])
        if len(agents_dedup) >= count:
            return agents_dedup
    raise ValueError("Not enough unique agents for given count")


def main():
    args = parse_args()
    set_seeds(args.seed)

    # Because I keep forgetting to select the correct env
    assert args.env in args.history_file

    env = create_env(args.env)
    print("Loading environment", args.env)
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    history = pickle_load(args.history_file)
    if args.best_agent:
        agents = get_best_agents_dedup(history, 1)
        eval_outfile = 'eval_best_agent'
    elif args.best_agent_ensemble:
        agents = get_best_agents_dedup(history, 3)
        eval_outfile = 'eval_best_agent_ensemble'
    elif args.last_agent:
        agents = [history['agents'][-1][0]]
        eval_outfile = 'eval_last_agent'
    elif args.last_agent_ensemble:
        agents = history['agents'][-1][:3]
        eval_outfile = 'eval_last_agent_ensemble'
    else:
        assert False

    dim_in, dim_out = get_input_output_dim(env)
    policies = [a.get_policy(dim_in, dim_out) for a in agents]
    policy = EnsemblePolicy(policies)

    if args.evaluate:
        total_rew = []
        for i in range(args.evaluate):
            env.seed(args.seed + i)
            ep_rew, ep_len = run_episode(env, policy)
            total_rew.append(ep_rew)
            print(f"ep = {i + 1}/{args.evaluate} reward = {ep_rew:.2f} len = {ep_len}")
        print(f"mean_reward = {np.mean(total_rew):.2f} +- {np.std(total_rew):.2f}")
        np.save(Path(args.history_file).with_name(eval_outfile), total_rew)

    elif args.render:
        for i in count():
            env.seed(args.seed + i)
            ep_rew, ep_len = run_episode(env, policy, render_human)
            print(f"ep = {i + 1}/inf reward = {ep_rew:.2f} len = {ep_len}")

    elif args.gif:
        render_gif = RenderGif()
        env.seed(args.seed)
        run_episode(env, policy, render_gif)
        render_gif.save(args.gif)


if __name__ == '__main__':
    main()
