#!/usr/bin/env python
'''Wrapper for pescador streams'''

import os
import numpy as np
import pandas as pd

import jams
import librosa

import pescador


def intersect_labels(annotation, time, duration, overlap):
    '''Slice a JAMS annotation frame down to an observation window.

    Parameters
    ----------
    annotation : jams.JamsFrame

    time : float > 0
    duration : float > 0
        The window, in seconds

    overlap : float > 0
        The minimum amount of (total) overlap time necessary
        to retain the label

    Returns
    -------
    labels : list
        A list of values corresponding to matching annotations
    '''

    a_trim = annotation.copy()

    # Clip intervals to the target
    a_trim['time'] = annotation['time'].clip(lower=pd.to_timedelta(time,
                                                                   unit='s'))
    a_trim['end'] = (annotation['time'] +
                     annotation['duration']).clip(upper=pd.to_timedelta(time +
                                                                        duration,
                                                                        unit='s'))

    a_trim['duration'] = a_trim['end'] - a_trim['time']

    # Select out the non-empty intervals
    a_trim = a_trim[a_trim['duration'] > 0]

    # If there are no labeled intervals left, return an empty list
    if not len(a_trim):
        return []

    # Aggregate time by label
    label_time = a_trim.groupby('value')['duration'].sum()

    # Keep only those that span enough time
    label_time = label_time[label_time >= pd.to_timedelta(overlap, unit='s')]

    return list(label_time.index)


def generate_data(name, data_path, label_encoder,
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

    data = np.load(featurefile)

    jamfile = os.path.join(data_path, '{}.jams'.format(name))

    jam = jams.load(jamfile)

    annotation = jam.annotations[0].data

    annotation = annotation[annotation['value'].isin(label_encoder.classes_)]

    duration = librosa.frames_to_time(n_columns)[0]
    n_total = data['C'].shape[1]

    while len(annotation):
        # Slice a patch
        idx = np.random.randint(0, n_total - n_columns)

        # Slice the labels
        y = intersect_labels(annotation,
                             librosa.frames_to_time(idx)[0],
                             duration,
                             min_overlap)

        yield dict(X=data['C'][:, idx:idx+n_columns].T[np.newaxis, np.newaxis],
                   Y=label_encoder.transform([y]))


def bufmux(batch_size, k,
           file_ids, aug_ids, data_path,
           label_encoder,
           lam=256.0,
           with_replacement=True,
           prune_empty_seeds=False,
           n_columns=128, min_overlap=0.25):
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
            fname = '{}_{:05d}'.format(file_id, aug_id)
            seeds.append(pescador.Streamer(generate_data,
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
