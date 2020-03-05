from glob import glob

import numpy as np

games = [
    'AtlantisNoFrameskip-v4',
    'FrostbiteNoFrameskip-v4',
    'GravitarNoFrameskip-v4',
    'KangarooNoFrameskip-v4',
    'SeaquestNoFrameskip-v4',
    'VentureNoFrameskip-v4',
]

variants = [
    ('v0', 'eval_best_agent'),
    ('v0', 'eval_best_agent_ensemble'),
    ('v0', 'eval_last_agent'),
    # ('v0', 'eval_last_agent_ensemble'),
    ('v1', 'eval_best_agent'),
    ('v1', 'eval_best_agent_ensemble'),
    ('v1', 'eval_last_agent'),
    # ('v1', 'eval_last_agent_ensemble'),
]

for game in games:
    for trainvar, evalvar in variants:
        f = glob(f'results/{game}/{trainvar}-*/{evalvar}.npy')
        if len(f) == 1:
            data = np.load(f[0])
            mean, std = np.mean(data), np.std(data)
            print(f"{game:30} {trainvar} {evalvar:>30} | {mean:8.1f} +- {std:8.1f}")
        else:
            print(f"{game:30} {trainvar} {evalvar:>30} | {'N/A':>20}")
    print()
