#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-03-02 19:05:34 by Brian McFee <brian.mcfee@nyu.edu>
'''Import onset data as JAMS files'''

import sys
import argparse
import os

import librosa
import pandas as pd
import jams

__curator__ = dict(name='Colin Raffel', email='craffel@gmail.com')
__corpus__ = 'Ballroom+Extra'


def parse_arguments(args):
    '''argument parsing'''
    parser = argparse.ArgumentParser(description='Ballroom++ parser')

    parser.add_argument('input_dir',
                        type=str,
                        help='Path to the data root directory')

    parser.add_argument('output_dir',
                        type=str,
                        help='Path to output jam files')

    return vars(parser.parse_args(args))


def get_metadata(infile):
    '''Construct a metadata object from flac data'''

    title = jams.util.filebase(infile)

    # Get the duration of the track
    y, sr = librosa.load(infile, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    # Format duration as time
    metadata = jams.FileMetadata(title=title,
                                 duration=duration)

    return metadata


def get_annotation(annfile):
    '''Get the onset time annotations'''

    curator = jams.Curator(**__curator__)

    metadata = jams.AnnotationMetadata(curator=curator, corpus=__corpus__)

    annotation = jams.Annotation('onsets', annotation_metadata=metadata)

    data = pd.read_table(annfile, header=None, sep='\s+')

    if len(data.columns) > 1:
        data = data.T

    for onset_time in data[0]:
        annotation.data.add_observation(time=onset_time,
                                        duration=0,
                                        value=None,
                                        confidence=None)

    return annotation


def save_jam(output_dir, jam):
    '''Save the output jam'''

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    outfile = os.extsep.join([jam.file_metadata.title, 'jams'])
    outfile = os.path.join(output_dir, outfile)

    print 'Saving {:s}'.format(outfile)
    jam.save(outfile)


def parse_data(input_dir, output_dir):
    '''Parse all the data in input_dir, save it in output_dir'''

    # Get a list of flac files
    flac_files = jams.util.find_with_extension(input_dir, 'flac')
    ann_files = [fname.replace('.flac', '.onsets') for fname in flac_files]

    for fname in ann_files:
        assert os.path.exists(fname)

    for flac, ann in zip(flac_files, ann_files):

        metadata = get_metadata(flac)

        onsets = get_annotation(ann)

        jam = jams.JAMS(file_metadata=metadata)
        jam.annotations.append(onsets)

        jam.sandbox.content_path = flac.replace(input_dir, '')

        save_jam(output_dir, jam)


if __name__ == '__main__':
    parameters = parse_arguments(sys.argv[1:])

    parse_data(**parameters)
