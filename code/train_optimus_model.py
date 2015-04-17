#!/usr/bin/env python
'''Train an instrument model'''

import argparse
import json
import numpy as np
import optimus
import sys
sys.path.insert(0, '/home/ejh333/src/optimus_dev/')
import os
import sklearn.preprocessing

import six
import ShuffleLabelsOut

from data_generator import bufmux

import optimus_models as models
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

NUM_FRAMES = 44
BATCH_SIZE = 50
DRIVER_ARGS = dict(
    max_iter=100000,
    save_freq=1000,
    print_freq=50)
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.02


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

    for _train, test in splitter:
        valsplitter = ShuffleLabelsOut.ShuffleLabelsOut(artist_ids[_train],
                                                        n_iter=1,
                                                        random_state=5)
        for train, val in valsplitter:
            train_file_ids = [track_names[_] for _ in train]
            val_file_ids = [track_names[_] for _ in val]
            test_file_ids = [track_names[_] for _ in test]

    # Save the train and test sets to disk
    tt_file = os.path.join(args.output_pattern, 'train_test.json')

    json.dump({'train': train_file_ids,
               'validation': val_file_ids,
               'test': test_file_ids},
              open(tt_file, 'w'),
              indent=2)

    # Create the generator; currently, at least, should yield dicts like
    #   dict(X=np.zeros([BATCH_SIZE, 1, NUM_FRAMES, NUM_FREQ_COEFFS]),
    #        Y=np.zeros([BATCH_SIZE, len(INSTRUMENTS)]))
    _stream = bufmux(
        BATCH_SIZE, 500, train_file_ids, aug_ids, args.input_path, LT,
        lam=128.0, with_replacement=True, n_columns=NUM_FRAMES,
        prune_empty_seeds=False, min_overlap=0.25)

    stream = pescador.zmq_stream(_stream, max_batches=DRIVER_ARGS['max_iter'])
    #print('Attempting to pull from stream')
    #print(next(stream))
    #print('success')
    # Build the two models:
    #  {loss, Z} = trainer(X, Y, learning_rate)
    #  {Z} = predictor(X)
    trainer, predictor = models.beastly_network(
        num_frames=NUM_FRAMES, num_classes=len(INSTRUMENTS),
        size=args.arch_size)

    # Init params, as needed.
    if args.init_param_file:
        print('Loading parameters: {0}'.format(args.init_param_file))
        trainer.load_param_values(args.init_param_file)

    # Wrap the trainer graph in a harness (checkpointing, logging, etc)
    print('Starting {0}'.format(args.name))
    driver = optimus.Driver(
        graph=trainer,
        name=args.name,
        output_directory=args.output_pattern)

    # Serialize the predictor graph.
    predictor_file = os.path.join(driver.output_directory, 'model_file.json')
    optimus.save(predictor, def_file=predictor_file)

    hyperparams = dict(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    driver.fit(stream, hyperparams=hyperparams, **DRIVER_ARGS)


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

    parser.add_argument('-s',
                        '--size',
                        dest='arch_size', type=str, default='large',
                        help='Size of the architecture')
    # Outputs
    parser.add_argument('-o',
                        '--output-pattern',
                        type=str,
                        dest='output_pattern',
                        required=True,
                        help='Pattern to store output models')

    parser.add_argument('-n',
                        '--name',
                        dest='name', type=str, default='deleteme',
                        help='Unique name for this training run.')

    parser.add_argument("--init_param_file",
                        metavar="init_param_file", type=str, default='',
                        help="Path to a NPZ archive for initialization the "
                        "parameters of the graph.")

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
