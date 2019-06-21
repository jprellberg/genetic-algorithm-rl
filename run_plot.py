import os.path
import numpy as np
import matplotlib.pyplot as plt

ROOT = 'results'
RUNS = [
    'frostbite',
]

for run in RUNS:
    path = os.path.join(ROOT, run, 'h_elite.npy')
    data = np.load(path)
    frames, mean, std = data[:, 0], data[:, 1], data[:, 2]
    plt.fill_between(frames, mean - std, mean + std, alpha=0.1)
    plt.plot(frames, mean, label=run)

plt.legend(loc='lower right')
plt.show()
