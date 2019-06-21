# Playing Atari with Genetic Algorithms

This repository is a reproduction of the ideas presented in [1] for learning to play Atari games using a genetic
algorithm.

An agent can be trained with `run_evo.py`. The most important command line arguments are:

* `--env <Atari gym environment name>`
* `--workers <number of parallel processes>`
* `--device <cpu or cuda>`

Progress can be visualized using `run_plot.py` and gameplay may be observed using `run_enjoy.py` or `run_gif.py`.
As an example, here is an agent that has been trained for 8.13e7 frames on the game Frostbite.

![Frostbite Gameplay](https://github.com/jprellberg/genetic-algorithm-rl/blob/master/results/frostbite/frostbite-gameplay.gif)

The following plot shows the evolution of episode rewards (averaged over 30 trials) of the population's
elite agent. Since the fitness evaluation is highly noisy the elite is sometimes replaced by an agent
with apparentely higher fitness that is actually worse.
 
![Frostbite Plots](https://github.com/jprellberg/genetic-algorithm-rl/blob/master/results/frostbite/forstbite-rewards.png)

The implementation is multiprocessing-based but not distributed and therefore can only be run on a single host.
GPUs can be leveraged but be aware that you may run into host memory issues given that each worker loads the CUDA
runtime. For example, I ran experiments using 20 worker processes that use 2 GPUs and just initializing the processes
takes about 50 GB of host memory. When using the CPU implementation this issue thankfully does not arise.

[1] Petroski Such, Felipe; Madhavan, Vashisht; Conti, Edoardo; Lehman, Joel; Stanley, Kenneth O.; Clune, Jeff:
Deep Neuroevolution: Genetic Algorithms are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning. eprint arXiv:1712.06567, 2017.