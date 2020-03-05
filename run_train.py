import argparse
import os
from collections import defaultdict

import numpy as np
import torch.multiprocessing

from atari import Agent
from population import Population
from utils import set_seeds, unique_string, get_logger, pickle_save, pickle_load
from worker import Worker


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', default='results')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--env', default='FrostbiteNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=int(np.random.randint(0, 100000)))
    parser.add_argument('--total-frames', type=int, default=int(1e9))
    parser.add_argument('--parents', type=int, default=50)
    parser.add_argument('--offspring', type=int, default=2000)
    parser.add_argument('--elites', type=int, default=1)
    parser.add_argument('--min-evals', type=int, default=1)
    parser.add_argument('--max-evals', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--eval-trials', type=int, default=50)
    parser.add_argument('--sigma', type=float, default=0.005)
    parser.add_argument('--max-rollout-len', type=int, default=3000)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seeds(args.seed)

    if os.path.exists(args.workdir):
        wdir = args.workdir
        os.chdir(wdir)
        history = pickle_load('history.pickle')
    else:
        wdir = os.path.join(args.workdir, args.env, args.tag + '-' + unique_string())
        os.makedirs(wdir, exist_ok=True)
        os.chdir(wdir)
        history = defaultdict(list)

    logger = get_logger()
    logger.info(wdir)
    logger.info(args)

    pop = Population(Agent, args.parents, args.offspring, args.elites, args.min_evals, args.max_evals)
    if 'frames' in history and 'agents' in history:
        pop.total_ep_len = history['frames'][-1]
        pop.agents = history['agents'][-1]

    workers = []
    logger.info(f"Starting {args.workers} worker processes")
    for i in range(args.workers):
        if args.device == 'cuda':
            device = torch.device('cuda', i % torch.cuda.device_count())
        else:
            device = torch.device('cpu')
        worker = Worker(pop.queue, args.env, args.sigma, args.max_rollout_len, device)
        worker.start()
        workers.append(worker)

    gen = 0
    logger.info("Starting evolution")
    while pop.total_ep_len < args.total_frames:
        pop.step()
        gen += 1

        logger.info(f"Generation {gen:04d}: frames={pop.total_ep_len:.2e}")
        logger.info(f"Pop: {[round(x.fitness(), 1) for x in pop.agents]}")

        fit_elite = pop.eval_elite(args.eval_trials)
        fit_elite_mean, fit_elite_std = np.mean(fit_elite), np.std(fit_elite)
        logger.info(f"Elite: {fit_elite_mean:.2f} +- {fit_elite_std:.2f}")

        # Save run data to file
        history['frames'].append(pop.total_ep_len)
        history['agents'].append(pop.agents)
        history['fit_elite'].append(fit_elite)
        history['fit_elite_mean'].append(fit_elite_mean)
        history['fit_elite_std'].append(fit_elite_std)
        pickle_save(history, 'history.pickle')

        assert all(w.is_alive() for w in workers)

    logger.info(f"Shutting down worker processes")
    pop.queue.stop_workers(args.workers)
    for w in workers:
        w.join()

    logger.info("Finished")


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
