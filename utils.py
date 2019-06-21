import datetime
import logging
import os
import pathlib
import random
import string
import sys

import torch
import numpy


def set_seeds(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model, grad_only=True):
    return sum(p.numel() for p in model.parameters() if not grad_only or p.requires_grad)


def unique_string():
    return '{}.{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%SZ'),
                          ''.join(random.choice(string.ascii_uppercase) for _ in range(4)))


def create_unique_dir(root=None):
    filename = unique_string()
    if root is not None:
        filename = os.path.join(root, filename)
    pathlib.Path(filename).mkdir(parents=True, exist_ok=True)
    return filename


def get_logger(output_file='log.txt'):
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S')

        # Create file handler
        if output_file is not None:
            fh = logging.FileHandler(output_file)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        # Create console handler
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
