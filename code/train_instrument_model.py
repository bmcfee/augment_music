#!/usr/bin/env python
'''Train an instrument model'''


import argparse
import sys

import os
import glob

import numpy as np
import pandas as pd

import pescador
import boyardeep

import data_generator


def process_args(args):

    parser = argparse.ArgumentParser(description="Train instrument models with muda pipelines")

    parser.add_argument('-o',
                        '--output-pattern',
                        type=str,
                        dest='output_pattern',
                        required=True,
                        help='Pattern to store output models')

    parser.add_argument('-i',
                        '--input-path',
                        dest='input_path',
                        type=str,
                        required=True,
                        help='Path to input data')

    parser.add_argument('-t',
                        '--training-index',
                        type=str,
                        dest='index_file',
                        required=True,
                        help='Path to file listing the valid augmentation samples for training')

    parser.add_argument('-m',
                        '--model-file',
                        dest='model_file',
                        type=str,
                        required=True,
                        help='Path to the model specification file')

    parser.add_argument('-s',
                        '--seed',
                        dest='rng',
                        type=int,
                        default=0x10101010,
                        help='Random number seed for train/test splits')

    parser.add_argument('--validation-tracks',
                        dest='n_validate_tracks',
                        type=int,
                        default=20,
                        help='Number of tracks to hold out for validation')

    parser.add_argument('--validation-samples',
                        dest='n_validate_samples',
                        type=int,
                        default=2000,
                        help='Number of samples to hold for validation')

    return vars(parser.parse_args(args))


if __name__ == '__main__':

    parameters = process_args(sys.argv[1:])
