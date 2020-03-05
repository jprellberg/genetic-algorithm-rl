import random

import torch.multiprocessing as mp


class Population:
    def __init__(self, generator, num_parents, num_offspring, num_elites, min_evals, max_evals):
        """
        :param generator: function that returns an agent
        :param num_parents: how many of the elites and best offspring are selected to be part
                            of the next generation
        :param num_offspring: how many agents are created from the parents
        :param num_elites: how many of the best agents are copied from the last generation
        :param min_evals: min. number of evaluations for all agents
        :param max_evals: max. number of evaluations for elites (and by extension all agents because
                          only elites ever get evaluated again after the initial evaluation)
        """
        super().__init__()
        self.queue = PopulationQueue()
        self.total_ep_len = 0

        self.num_parents = num_parents
        self.num_offspring = num_offspring
        self.num_elites = num_elites
        self.min_evals = min_evals
        self.max_evals = max_evals
        self.agents = [generator() for _ in range(self.num_offspring)]

    def step(self):
        if self.total_ep_len == 0:
            # Evaluate initial population
            offspring = self._submit_tasks(self.agents, mutate=False, reset=True)
            # Re-evaluate the minimum number of times
            for i in range(self.min_evals - 1):
                offspring = self._submit_tasks(offspring, mutate=False, reset=False)
            elites = []
        else:
            # Randomly choose parents, mutate and evaluate
            parents = random.choices(self.agents, k=self.num_offspring)
            offspring = self._submit_tasks(parents, mutate=True, reset=True)
            # Re-evaluate the minimum number of times
            for i in range(self.min_evals - 1):
                offspring = self._submit_tasks(offspring, mutate=False, reset=False)
            elites = self.agents[:self.num_elites]
            # Reevaluate elites for more accurate fitness up to max_evals times
            finished_elites = [e for e in elites if e.num_evals() >= self.max_evals]
            reeval_elites = [e for e in elites if e.num_evals() < self.max_evals]
            reeval_elites = self._submit_tasks(reeval_elites, mutate=False, reset=False)
            elites = finished_elites + reeval_elites

        # Truncation selection
        offspring = sorted(offspring, key=lambda p: p.fitness(), reverse=True)
        new_parents = offspring[:self.num_parents - len(elites)]
        self.agents = sorted(elites + new_parents, key=lambda p: p.fitness(), reverse=True)

    def _submit_tasks(self, agents, **kwargs):
        # Import here because it sets the multiprocessing start method
        from tqdm import trange
        # Submit tasks to worker queue
        for agent in agents:
            self.queue.master_put((agent, kwargs))
        # Collect results as soon as they are available
        results = []
        for _ in trange(len(agents), leave=False):
            item = self.queue.master_get()
            self.total_ep_len += item.ep_len[-1]
            results.append(item)
        return results

    def get_elite(self):
        return self.agents[0]

    def eval_elite(self, trials):
        from tqdm import trange
        fit = []
        for _ in range(trials):
            self.queue.master_put((self.get_elite(), {'mutate': False, 'reset': True}))
        for _ in trange(trials, leave=False):
            item = self.queue.master_get()
            fit.append(item.fitness())
        return fit


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
