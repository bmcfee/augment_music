#!/usr/bin/env python
'''Feature extraction'''

from __future__ import print_function

from argparse import ArgumentParser

import os
import sys
import librosa
import numpy as np

from joblib import Parallel, delayed

RESOLUTION = 3
N_BINS = 72 * RESOLUTION
BINS_PER_OCTAVE = 12 * RESOLUTION
FMIN = librosa.note_to_hz('C3')


def get_output_name(output_dir, fname):

    root = os.path.splitext(os.path.basename(fname))[0]

    return os.path.join(output_dir, os.path.extsep.join([root, 'npz']))


def compute_features(fname):
    y, sr = librosa.load(fname)

    C = librosa.cqt(y, sr=sr,
                    fmin=FMIN,
                    n_bins=N_BINS,
                    bins_per_octave=BINS_PER_OCTAVE).astype(np.float32)

    return C


def feature_extractor(fname, output_dir):

    C = compute_features(fname)

    output_name = get_output_name(output_dir, fname)

    np.savez(output_name, C=C)


def run(ogg_files=None, verbose=0, output_dir=None, num_jobs=2):
    '''Do the needful'''

    # Step 3: schedule the jobs
    feature = delayed(feature_extractor)

    Parallel(n_jobs=num_jobs,
             verbose=verbose)(feature(aud, output_dir)
                              for aud in ogg_files)


def get_params(args):
    '''Process command-line arguments'''

    parser = ArgumentParser(description='Extract features for classification')

    parser.add_argument('-n', '--num_jobs', dest='num_jobs',
                        type=int, default=1, help='Number of parallel threads')

    parser.add_argument('-v', '--verbose', type=int, default=0,
                        help='Verbosity')

    parser.add_argument('output_dir', type=str,
                        help='Path to store the output files')

    parser.add_argument('ogg_files', type=str, nargs='+',
                        help='Path to the annotations')

    return vars(parser.parse_args(args))


if __name__ == '__main__':
    params = get_params(sys.argv[1:])

    run(**params)
