import argparse
import os

import numpy as np
import torch.multiprocessing

from atari import create_random_agent
from population import Population
from utils import set_seeds, create_unique_dir, get_logger
from worker import Worker


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', default='results')
    parser.add_argument('--env', default='FrostbiteNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--total-frames', type=int, default=int(3e8))
    parser.add_argument('--parents', type=int, default=50)
    parser.add_argument('--offspring', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--eval-trials', type=int, default=30)
    parser.add_argument('--sigma', type=float, default=0.005)
    parser.add_argument('--max-rollout-len', type=int, default=5000)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    wdir = create_unique_dir(args.workdir)
    os.chdir(wdir)
    set_seeds(args.seed)

    logger = get_logger()
    logger.info(wdir)
    logger.info(args)

    pop = Population(lambda: create_random_agent(args.env), args.parents, args.offspring)

    workers = []
    logger.info(f"Starting {args.workers} worker processes")
    for i in range(args.workers):
        worker = Worker(pop.queue, args.env, args.seed + i, args.sigma,
                        args.max_rollout_len, args.device)
        worker.start()
        workers.append(worker)

    gen = 0
    h_elite_fit = []
    h_pop_fit = []
    logger.info("Starting evolution")
    while pop.total_ep_len < args.total_frames:
        pop.step()
        gen += 1

        logger.info(f"Generation {gen:04d}: frames={pop.total_ep_len:.2e}")
        logger.info(f"Pop: {[round(x.fitness()) for x in pop.items if not np.isneginf(x.fitness())]}")

        elite = pop.get_elite()
        torch.save(elite.state_dict(), f'agent_gen{gen:04d}.pt')

        elite_mean, elite_std = pop.eval_elite(args.eval_trials)
        logger.info(f"Elite: {elite_mean:.2f} +- {elite_std:.2f}")

        # h_pop_fit contains total frames and the fitness of each population member for every generation
        h_pop_fit.append([pop.total_ep_len] + [x.fitness() for x in pop.items])
        # h_elite_fit contains total frames and fitness mean+standard deviation of the elite
        # evaluated args.eval_trial times for every generation
        h_elite_fit.append((pop.total_ep_len, elite_mean, elite_std))
        np.save('h_pop.npy', h_pop_fit)
        np.save('h_elite.npy', h_elite_fit)

        assert all(w.is_alive() for w in workers)

    logger.info(f"Shutting down worker processes")
    pop.queue.stop_workers(args.workers)
    for w in workers:
        w.join()

    logger.info("Finished")


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
