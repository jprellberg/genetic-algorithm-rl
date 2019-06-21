from copy import deepcopy

import torch
import torch.multiprocessing as mp

from atari import create_env


class Worker(mp.Process):
    def __init__(self, popqueue, env_id, seed, sigma, max_rollout_len, device):
        super().__init__(daemon=True)
        self.popqueue = popqueue
        self.env_id = env_id
        self.seed = seed
        self.sigma = sigma
        self.max_rollout_len = max_rollout_len
        self.device = device

    def run(self):
        torch.set_num_threads(1)
        env = create_env(self.env_id)
        env.seed(self.seed)
        while True:
            item = self.popqueue.worker_get()
            if item is None:
                return

            if self.device == 'cuda':
                device = torch.device('cuda', self.seed % torch.cuda.device_count())
            else:
                device = torch.device('cpu')

            off, mutate = item
            off = deepcopy(off)
            # Convert to float32 before mutating because the operations aren't available
            # on CPU for half-tensors
            off.to(torch.float32)
            if mutate:
                # Mutate also resets statistics
                off.mutate(self.sigma)
            else:
                off.reset_stats()
            off.to(device)

            ep_rew = 0
            ep_len = 0

            obs = env.reset()
            for i in range(self.max_rollout_len):
                a = off(obs, device)
                obs, rew, done, _ = env.step(a)
                ep_rew += rew
                ep_len += 1
                if done:
                    break

            off.ep_rew = ep_rew
            off.ep_len = ep_len

            # Move to CPU to free GPU memory
            off.cpu()
            # Convert to half to conserve host memory
            off.half()
            self.popqueue.worker_put(off)
