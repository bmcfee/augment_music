#!/usr/bin/env python
'''Train an instrument model'''


import argparse
import sys

import json
import numpy as np

import pescador
import sklearn.preprocessing
import boyardeep

from data_generator import bufmux

from ShuffleLabelsOut import ShuffleLabelsOut


# The well represented instruments, as listed on the medleydb page
INSTRUMENTS = ['drum set',
               'electric bass',
               'piano',
               'male singer',
               'clean electric guitar',
               'vocalists',
               'synthesizer',
               'female singer',
               'acoustic guitar',
               'distorted electric guitar',
               'auxiliary percussion',
               'double bass',
               'violin',
               'cello',
               'flute',
               'mandolin']


def process_args(args):

    parser = argparse.ArgumentParser(description="Train instrument models "
                                                 "with muda pipelines")

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
                        help='Path to file listing the valid augmentation '
                             'samples for training')

    parser.add_argument('-a',
                        '--artist-index',
                        type=str,
                        dest='artist_file',
                        required=True,
                        help='Path to the file containing the '
                             'track->artist index')

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


def load_artists(artist_file):
    '''Load the artist mapping'''

    data = json.load(open(artist_file, 'r'))

    track_names = []
    artist_ids = []

    for track, artist in sorted(data.items()):
        track_names.append(track)
        artist_ids.append(artist)

    return track_names, np.asarray(artist_ids)


def run_experiment(output_pattern=None,
                   input_path=None,
                   index_file=None,
                   artist_file=None,
                   model_file=None,
                   rng=None,
                   n_validate_tracks=None,
                   n_validate_samples=None,
                   batch_size=512,
                   k=500,
                   lam=256.0,
                   with_replacement=True,
                   prune_empty_seeds=False,
                   min_overlap=0.25):

    track_names, artist_ids = load_artists(artist_file)
    aug_ids = np.loadtxt(index_file, dtype=int)

    splitter = ShuffleLabelsOut(artist_ids, random_state=rng)

    LT = sklearn.preprocessing.MultiLabelBinarizer(classes=INSTRUMENTS)
    LT.fit(INSTRUMENTS)

    arch = boyardeep.load_architecture(model_file)

    # Load the input size from the model spec's input layer
    n_columns = arch['layers'][0][-1]['kwargs']['shape'][-1]

    for _train, test in splitter:
        vsplitter = ShuffleLabelsOut(artist_ids[_train],
                                     n_iter=1,
                                     test_size=n_validate_tracks,
                                     random_state=rng)

        for train, valid in vsplitter:

            # Make a training stream
            train_stream = bufmux(batch_size,
                                  k,
                                  [track_names[_] for _ in train],
                                  aug_ids,
                                  input_path,
                                  LT,
                                  lam=lam,
                                  with_replacement=with_replacement,
                                  prune_empty_seeds=prune_empty_seeds,
                                  n_columns=n_columns,
                                  min_overlap=min_overlap)

            # Wrap the training stream in a separate thread
            zts = pescador.zmq_stream(train_stream)

            # Build the model
            estimator = boyardeep.Boyardeep(arch,
                                            multilabel=True,
                                            regression=False)

            model = pescador.StreamLearner(estimator)

            model.iter_fit(zts)
            
            #TODO:  2015-04-13 21:39:22 by Brian McFee <brian.mcfee@nyu.edu>
            # Validation set
            # stopping criteria
            # incremental checkpoints


if __name__ == '__main__':

    parameters = process_args(sys.argv[1:])
