import itertools
import subprocess
import sys
from glob import glob
from pathlib import Path


def configs():
    games = [
        'AtlantisNoFrameskip-v4',
        'FrostbiteNoFrameskip-v4',
        'GravitarNoFrameskip-v4',
        'KangarooNoFrameskip-v4',
        'SeaquestNoFrameskip-v4',
        'VentureNoFrameskip-v4',
    ]
    cfgs = []
    for game in games:
        for hf in glob(f'results/{game}/*/history.pickle'):
            cfg1 = config_product(env=game,
                                  history_file=hf,
                                  best_agent=None,
                                  evaluate=100)
            cfg2 = config_product(env=game,
                                  history_file=hf,
                                  best_agent_ensemble=None,
                                  evaluate=100)
            cfg3 = config_product(env=game,
                                  history_file=hf,
                                  best_agent=None,
                                  gif=Path(hf).with_name('best_agent.gif'))
            cfgs.append(cfg1)
            cfgs.append(cfg2)
            # cfgs.append(cfg3)
    return itertools.chain(*cfgs)


def config_product(**kwargs):
    keys = list(kwargs.keys())
    vals = list(kwargs.values())
    vals = [x if isinstance(x, list) else [x] for x in vals]
    for item in itertools.product(*vals):
        yield dict(zip(keys, item))


def launch_call(calls, id):
    call = list(calls)[id]
    print("Launching:", ' '.join(call))
    subprocess.run(call)


def list_calls(calls):
    for i, call in enumerate(calls):
        print(i, ' '.join(call), sep='\t')


def to_call(cfg):
    call = ['python3.6', '-u', 'run_eval.py']
    for k, v in cfg.items():
        k = k.replace('_', '-')
        if v is not None:
            call += [f'--{k}', str(v)]
        else:
            call += [f'--{k}']
    return call


def launcher(configs):
    calls = [to_call(cfg) for cfg in configs]
    if len(sys.argv) > 1:
        id = int(sys.argv[1])
        launch_call(calls, id)
    else:
        print("Expected task ID as argument")
        list_calls(calls)


if __name__ == '__main__':
    launcher(configs())