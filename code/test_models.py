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
                        help='Path to best-parameter json file')

    return vars(parser.parse_argus(args))


def run(best_params=None):
    '''Run the tests'''

    best_parameters = json.load(open(best_params, 'r'))

    for aug_idx in best_parameters:
        for fold in best_parameters[aug_idx]:
            parameters = best_parameters[aug_idx][fold]

            os.environ['param_file'] = parameters

            subprocess.check_call(['./experiment.sh',
                                   aug_idx,
                                   'large',
                                   fold,
                                   'evaluate'])


if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])
    run(**params)
