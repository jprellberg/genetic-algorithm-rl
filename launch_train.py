import itertools
import subprocess
import sys


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
    # Use a loop instead of config_products list functionality to have v0 and v1 in alternating order
    # for the same game
    for game in games:
        cfgs += [config_product(env=game,
                                parents=10,
                                offspring=5000,
                                elites=1,
                                min_evals=1,
                                max_evals=1,
                                workers=30,
                                tag='v0'),

                 config_product(env=game,
                                parents=100,
                                offspring=2000,
                                elites=40,
                                min_evals=1,
                                max_evals=20,
                                workers=30,
                                tag='v1'),
                 ]
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
    call = ['python3.6', '-u', 'run_train.py']
    for k, v in cfg.items():
        k = k.replace('_', '-')
        call += [f'--{k}', str(v)]
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