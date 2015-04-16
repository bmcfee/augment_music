#!/usr/bin/env python
'''Train an instrument model'''

import argparse
import json
import numpy as np
import sklearn.preprocessing

import cPickle as pickle
import ShuffleLabelsOut

from data_generator import bufmux

import pescador


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

NUM_FRAMES = 128
BATCH_SIZE = 50
DRIVER_ARGS = dict(
    max_iter=100000,
    save_freq=1000,
    print_freq=50)
LEARNING_RATE = 0.01


def load_artists(artist_file):
    '''Load the artist mapping'''

    data = json.load(open(artist_file, 'r'))

    track_names = []
    artist_ids = []

    for track, artist in sorted(data.items()):
        track_names.append(track)
        artist_ids.append(artist)

    return track_names, np.asarray(artist_ids)


def main(args):

    track_names, artist_ids = load_artists(args.artist_file)
    aug_ids = np.atleast_1d(np.loadtxt(args.index_file, dtype=int))

    LT = sklearn.preprocessing.MultiLabelBinarizer(classes=INSTRUMENTS)
    LT.fit(INSTRUMENTS)

    # TODO(ejhumphrey): I don't know what goes here.
    splitter = ShuffleLabelsOut.ShuffleLabelsOut(artist_ids,
                                                 n_iter=1,
                                                 random_state=5)

    for train, test in splitter:
        pass

    file_ids = [track_names[_] for _ in train]

    # Create the generator; currently, at least, should yield dicts like
    #   dict(X=np.zeros([BATCH_SIZE, 1, NUM_FRAMES, NUM_FREQ_COEFFS]),
    #        Y=np.zeros([BATCH_SIZE, len(INSTRUMENTS)]))
    _stream = bufmux(
        BATCH_SIZE, 500, file_ids, aug_ids, args.input_path, LT,
        lam=256.0, with_replacement=True, n_columns=NUM_FRAMES,
        prune_empty_seeds=False, min_overlap=0.25)

    stream = pescador.zmq_stream(_stream, max_batches=DRIVER_ARGS['max_iter'])

    data = []
    for batch in stream:
        data.append(batch)
        if len(data) > 10 * BATCH_SIZE:
            break

    with open(args.output, 'w') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train instrument models "
                                                 "with muda pipelines")
    # Inputs
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

    # Outputs
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        dest='output',
                        required=True,
                        help='Pattern to store output data')

    # Recco: Split data external to the training script.
    # --------
    # parser.add_argument('-s',
    #                     '--seed',
    #                     dest='rng',
    #                     type=int,
    #                     default=0x10101010,
    #                     help='Random number seed for train/test splits')

    # parser.add_argument('--validation-tracks',
    #                     dest='n_validate_tracks',
    #                     type=int,
    #                     default=20,
    #                     help='Number of tracks to hold out for validation')

    # parser.add_argument('--validation-samples',
    #                     dest='n_validate_samples',
    #                     type=int,
    #                     default=2000,
    #                     help='Number of samples to hold for validation')

    main(parser.parse_args())
