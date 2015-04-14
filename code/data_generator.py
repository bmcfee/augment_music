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


def generate_data(name, feature_path, jam_path, label_encoder,
                  n_columns=128, min_overlap=0.25):
    '''Data generator for a single track

    Parameters
    ----------
    name : str
        The name of the track, eg, 'MusicDelta_Reggae_000081'

    feature_path : str
        Path to the directory containing feature files

    jam_path : str
        Path to the directory containing jams files

    label_encoder : sklearn.preprocessing.MultiLabelBinarizer
        Transform label arrays into binary vectors

    n_columns : int > 0
        The width of patches to sample

    min_overlap : float > 0
        The minimum overlap required for a valid observation
    '''

    featurefile = os.path.join(feature_path, '{}_MIX.npz'.format(name))

    data = np.load(featurefile)

    jamfile = os.path.join(jam_path, '{}.jams'.format(name))

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

        yield dict(X=data['C'][:, idx:idx+n_columns][np.newaxis, np.newaxis],
                   Y=label_encoder.transform([y]))
