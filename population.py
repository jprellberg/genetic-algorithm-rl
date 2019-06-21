import random

import numpy as np
import torch.multiprocessing as mp


class Population:
    """
    Population model that follows the UBER Atari paper, implementing elitism and
    truncation selection.
    """
    def __init__(self, generator, num_parents, num_offspring):
        super().__init__()
        self.queue = PopulationQueue()
        self.total_ep_len = 0

        self.num_parents = num_parents
        self.num_offspring = num_offspring
        self.items = [generator() for _ in range(self.num_parents)]

    def step(self):
        # Import here because it sets the multiprocessing start method
        from tqdm import trange

        # Mutation
        for i in range(self.num_offspring):
            choice = random.choice(self.items)
            self.queue.master_put((choice, True))

        # Elitism
        offspring = [self.items[0]]
        for _ in trange(self.num_offspring, leave=False):
            item = self.queue.master_get()
            self.total_ep_len += item.ep_len
            offspring.append(item)

        # Truncation selection
        offspring = sorted(offspring, key=lambda p: p.fitness(), reverse=True)
        self.items = offspring[:len(self.items)]

    def get_elite(self):
        return self.items[0]

    def eval_elite(self, trials):
        from tqdm import trange
        fit = []
        for _ in range(trials):
            self.queue.master_put((self.get_elite(), False))
        for _ in trange(trials, leave=False):
            item = self.queue.master_get()
            fit.append(item.fitness())

        mean, std = np.mean(fit), np.std(fit)
        return mean, std


class PopulationQueue:
    def __init__(self, maxsize=0):
        self.m2w = mp.Queue(maxsize=maxsize)
        self.w2m = mp.Queue(maxsize=maxsize)

    def stop_workers(self, count):
        for i in range(count):
            self.master_put(None)

    def master_put(self, *args, **kwargs):
        self.m2w.put(*args, **kwargs)

    def master_get(self, *args, **kwargs):
        return self.w2m.get(*args, **kwargs)

    def worker_put(self, *args, **kwargs):
        self.w2m.put(*args, **kwargs)

    def worker_get(self, *args, **kwargs):
        return self.m2w.get(*args, **kwargs)
