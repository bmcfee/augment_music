#!/usr/bin/env python
'''script to run experiment tests'''
import argparse
import sys
import subprocess
import os
import json


def process_arguments(args):
    '''Argument parser'''
    parser = argparse.ArgumentParser(description='Test computer')

    parser.add_argument('best_params', type=str,
                        help='Relative path to best-parameter json file')

    parser.add_argument('base_dir', type=str,
                        help='Path to the results')

    return vars(parser.parse_args(args))


def run(best_params=None, base_dir=None):
    '''Run the tests'''

    best_parameters = json.load(open(best_params, 'r'))

    for aug_idx in best_parameters:
        for fold in best_parameters[aug_idx]:
            parameters = os.path.join(base_dir, best_parameters[aug_idx][fold])

            os.environ['param_file'] = parameters

            subprocess.check_call(['./experiment.sh',
                                   aug_idx,
                                   'large',
                                   fold,
                                   'evaluate'])


if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])
    run(**params)
