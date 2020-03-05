from copy import deepcopy

import torch
import torch.multiprocessing as mp

from atari import create_env, get_input_output_dim


class Worker(mp.Process):
    def __init__(self, popqueue, env_id, sigma, max_rollout_len, device):
        super().__init__(daemon=True)
        self.popqueue = popqueue
        self.env_id = env_id
        self.sigma = sigma
        self.max_rollout_len = max_rollout_len
        self.device = device

    def run(self):
        torch.set_num_threads(1)
        env = create_env(self.env_id)
        dim_in, dim_out = get_input_output_dim(env)
        while True:
            item = self.popqueue.worker_get()
            if item is None:
                return

            off, args = item
            off = deepcopy(off)
            if args['mutate']:
                off.mutate(self.sigma)
                # mutate also calls reset_stats
            if args['reset']:
                off.reset_stats()

            policy = off.get_policy(dim_in, dim_out).to(self.device)
            ep_rew = 0
            ep_len = 0

            env.seed()
            obs = env.reset()
            for i in range(self.max_rollout_len):
                a = policy(obs, self.device).argmax().item()
                obs, rew, done, _ = env.step(a)
                ep_rew += rew
                ep_len += 1
                if done:
                    break

            off.ep_rew.append(ep_rew)
            off.ep_len.append(ep_len)

            self.popqueue.worker_put(off)
