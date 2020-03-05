import os.path
import numpy as np
import matplotlib.pyplot as plt

from utils import pickle_load


EXPERIMENTS = {
    'Atlantis': {
        'gridpos': (0, 0),
        'v0': 'results/AtlantisNoFrameskip-v4/v0-20200207T093806Z.XVUC',
        'v1': 'results/AtlantisNoFrameskip-v4/v1-20200208T004543Z.XYVW',
    },
    'Frostbite': {
        'gridpos': (0, 1),
        'v0': 'results/FrostbiteNoFrameskip-v4/v0-20200210T135157Z.CAMY',
        'v1': 'results/FrostbiteNoFrameskip-v4/v1-20200211T122208Z.NLHZ',
    },
    'Gravitar': {
        'gridpos': (1, 0),
        'v0': 'results/GravitarNoFrameskip-v4/v0-20200214T135209Z.TJTZ',
        'v1': 'results/GravitarNoFrameskip-v4/v1-20200215T122214Z.ZZOF',
    },
    'Kangaroo': {
        'gridpos': (1, 1),
        'v0': 'results/KangarooNoFrameskip-v4/v0-20200225T043120Z.MJOH',
        'v1': 'results/KangarooNoFrameskip-v4/v1-20200227T070500Z.GVIF',
    },
    'Seaquest': {
        'gridpos': (2, 0),
        'v0': 'results/SeaquestNoFrameskip-v4/v0-20200218T204430Z.FBTS',
        'v1': 'results/SeaquestNoFrameskip-v4/v1-20200220T141030Z.WUTD',
    },
    'Venture': {
        'gridpos': (2, 1),
        'v0': 'results/VentureNoFrameskip-v4/v0-20200222T060036Z.UNXG',
        'v1': 'results/VentureNoFrameskip-v4/v1-20200224T024148Z.SSUV',
    }
}


def plot_run(ax, run):
    data = pickle_load(os.path.join(run, 'history.pickle'))
    frames = data['frames']
    fit_elite_mean = np.array(data['fit_elite_mean'])
    fit_elite_std = np.array(data['fit_elite_std'])

    # Get improvements to mean fitness
    mean_max = np.maximum.accumulate(fit_elite_mean)

    # Get the std values corresponding to changes in the maximum of the mean
    ind = np.flatnonzero(fit_elite_mean == mean_max)
    std_max = np.zeros_like(mean_max).astype(np.int64)
    std_max[ind] = ind
    std_max = np.maximum.accumulate(std_max)
    std_max = fit_elite_std[std_max]

    ax.fill_between(frames, mean_max - std_max, mean_max + std_max, alpha=0.1)
    return ax.plot(frames, mean_max, label=run)[0]


fig, axarr = plt.subplots(3, 2, figsize=(8, 12))
for game, data in EXPERIMENTS.items():
    gp = data['gridpos']
    axarr[gp].set_title(game)
    h0 = plot_run(axarr[gp], data['v0'])
    h1 = plot_run(axarr[gp], data['v1'])

for i in range(3):
    axarr[(i, 0)].set_ylabel("Score")
for i in range(2):
    axarr[(2, i)].set_xlabel("Frames")

fig.legend([h0, h1], ['Baseline', 'Proposal'],
           ncol=2,
           loc='lower center')

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.show()
