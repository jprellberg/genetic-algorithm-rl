# Playing Atari with Genetic Algorithms

This repository is a reproduction of the ideas presented in [1] for learning to play Atari games using a genetic
algorithm. The algorithm is extended with a re-evaluation procedure to counteract the strong fitness noise stemming
from non-determinism in the Atari environment.

An agent can be trained with `run_train.py` and results can be visualized with `run_plot.py`. Use `run_eval.py` to
perform further evaluation of an agent or to observe the gameplay.

As an example, here is an agent that has been trained for 1e9 frames on the game Frostbite using the proposed
extended algorithm with re-evaluations and new hyperparameters.

![Frostbite Gameplay](https://github.com/jprellberg/genetic-algorithm-rl/blob/master/results_frostbite.gif)

The following plot shows the evolution of episode rewards of the population's elite agent. It compares the baseline
algorithm from [1] to the proposed algorithm.
 
![Frostbite Plots](https://github.com/jprellberg/genetic-algorithm-rl/blob/master/results.png)

The implementation is multiprocessing-based but not distributed and therefore can only be run on a single host.
GPUs can be leveraged but be aware that you may run into host memory issues given that each worker loads the CUDA
runtime. For example, I ran experiments using 20 worker processes that use 2 GPUs and just initializing the processes
takes about 50 GB of host memory. When using the CPU implementation this issue thankfully does not arise.

[1] Petroski Such, Felipe; Madhavan, Vashisht; Conti, Edoardo; Lehman, Joel; Stanley, Kenneth O.; Clune, Jeff:
Deep Neuroevolution: Genetic Algorithms are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning. eprint arXiv:1712.06567, 2017.
