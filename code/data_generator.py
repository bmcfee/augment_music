#!/usr/bin/env python
'''Wrapper for pescador streams'''

import os
import numpy as np
import librosa

import jams

import pescador

from extract_features import compute_features


def frames_to_time(frames, hop_length=512, sr=22050):

    return np.atleast_1d(frames) * hop_length / float(sr)


def time_to_frames(times, hop_length=512, sr=22050):

    return (np.atleast_1d(times) * float(sr) / hop_length).astype(int)


def make_label_matrix(n, ann, label_encoder):
    '''Generate a binary matrix of labels, sampled to match X

    Parameters
    ----------
    n : int > 0
        time extent of the labels

    ann : jamsframe
        the annotation array

    label_encoder : sklearn.preprocessing.MultiLabelBinarizer
        The label encoder

    Returns
    -------
    Y : np.ndarray [shape=(n_classes, n)]
    '''

    n_classes = len(label_encoder.classes_)

    Y = np.zeros((n_classes, n))

    time = time_to_frames(ann['time'] / np.timedelta64(1, 's'))
    duration = time_to_frames(ann['duration'] / np.timedelta64(1, 's'))
    index = label_encoder.transform([[x] for x in ann['value'].values])

    for i, j, (_, label) in zip(time, duration, np.argwhere(index)):
        Y[label, i:i+j] = 1

    return Y


def makexy(name, data_path, label_encoder,
           n_columns=128, min_overlap=0.25):
    '''Data generator for a single track

    Parameters
    ----------
    name : str
        The name of the track, eg, 'MusicDelta_Reggae_000081'

    data_path : str
        Path to the directory containing feature and annotation files

    label_encoder : sklearn.preprocessing.MultiLabelBinarizer
        Transform label arrays into binary vectors

    n_columns : int > 0
        The width of patches to sample

    min_overlap : float > 0
        The minimum overlap required for a valid observation
    '''

    featurefile = os.path.join(data_path, '{}.npz'.format(name))

    X = librosa.logamplitude(np.load(featurefile)['C'])

    n_total = X.shape[1]

    jamfile = os.path.join(data_path, '{}.jams'.format(name))
    jam = jams.load(jamfile, validate=False)
    annotation = jam.annotations[0].data
    annotation = annotation[annotation['value'].isin(label_encoder.classes_)]

    Y = make_label_matrix(n_total, annotation, label_encoder)

    n_overlap = time_to_frames(min_overlap)

    return X, Y, n_total, n_overlap


def stream_data(name, data_path, label_encoder,
                n_columns=128, min_overlap=0.25):

    X, Y, n_total, n_overlap = makexy(name, data_path, label_encoder,
                                      n_columns=n_columns,
                                      min_overlap=min_overlap)

    for idx in np.arange(0, n_total - n_columns, n_columns):
        Xsamp = X[:, idx:idx+n_columns].T[np.newaxis, np.newaxis]
        Ysamp = (Y[:, idx:idx+n_columns].T.sum(axis=0, keepdims=True) >=
                 n_overlap)

        yield dict(X=Xsamp, Y=Ysamp.astype(int))


def sample_data(name, data_path, label_encoder,
                n_columns=128, min_overlap=0.25):

    X, Y, n_total, n_overlap = makexy(name, data_path, label_encoder,
                                      n_columns=n_columns,
                                      min_overlap=min_overlap)
    while True:
        # Slice a patch
        idx = np.random.randint(0, n_total - n_columns)

        Xsamp = X[:, idx:idx+n_columns].T[np.newaxis, np.newaxis]
        Ysamp = (Y[:, idx:idx+n_columns].T.sum(axis=0, keepdims=True) >=
                 n_overlap)

        yield dict(X=Xsamp, Y=Ysamp.astype(int))


def augment_file_id(file_id, aug_id):
    '''Return the augmented file id

    Parameters
    ----------
    file_id : str
        The prefix of the file

    aug_id : int
        The index of the augmentation

    Returns
    -------
    file_id_aug
        The augmentation-friendly formatted file id
    '''
    return '{}_{:05d}'.format(file_id, aug_id)


def make_unlabeled(name):
    '''Data generator for a single track

    Parameters
    ----------
    name : str
        The path to an audio file on disk
    '''

    C = compute_features(name)

    X = librosa.logamplitude(C)

    n_total = X.shape[1]

    return X, n_total


def stream_unlabeled(name, label_encoder, n_columns=128):

    X, n_total = make_unlabeled(name)

    Y = np.zeros((n_columns, len(label_encoder.classes_)), dtype=int)

    for idx in np.arange(0, n_total - n_columns, n_columns):
        Xsamp = X[:, idx:idx+n_columns].T[np.newaxis, np.newaxis]

        yield dict(X=Xsamp, Y=Y)


def bufmux(batch_size, k,
           file_ids, aug_ids, data_path,
           label_encoder,
           lam=256.0,
           with_replacement=True,
           prune_empty_seeds=False,
           n_columns=128,
           min_overlap=0.25):
    '''Make a parallel, multiplexed, pescador stream

    Parameters
    ----------
    batch_size : int > 0
        The number of examples to generate per batch

    k : int > 0
        the number of seeds to keep alive at any time

    file_ids : list of str
        filename prefixes to use, eg, 'MusicDelta_Reggae'

    aug_ids : list of int
        augmentation suffices to use, eg, `[48]`

    data_path : str
        Path to the directory containing npz,jams files on disk

    label_encoder : sklearn.preprocessing.MultiLabelBinarizer
        The multi-label encoder object

    lam : float > 0
    with_replacement : bool
    prune_empty_seeds : bool
        Lambda parameter for poisson variables in stream multiplexing
        Additional sampling parameters for `pescador.mux`

    n_columns : int > 0
        Width of patches to generate

    min_overlap : float > 0
        minimum degree of overlap (in seconds) to require for
        label/window overlap


    Returns
    -------
    streamer : pescador.Streamer
        A buffered multiplexing streamer
    '''

    seeds = []
    for file_id in file_ids:
        for aug_id in aug_ids:
            fname = augment_file_id(file_id, aug_id)
            seeds.append(pescador.Streamer(sample_data,
                                           fname,
                                           data_path,
                                           label_encoder,
                                           n_columns=n_columns,
                                           min_overlap=min_overlap))

    mux_streamer = pescador.Streamer(pescador.mux,
                                     seeds,
                                     None,
                                     k,
                                     lam=lam,
                                     with_replacement=with_replacement,
                                     prune_empty_seeds=prune_empty_seeds)

    return pescador.Streamer(pescador.buffer_streamer,
                             mux_streamer,
                             batch_size)
