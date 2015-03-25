#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#CREATED:2015-03-16 17:13:23 by Brian McFee <brian.mcfee@nyu.edu>
'''Parse MedleyDB annotations into JAMS format.'''

import sys
import argparse
import os

import librosa
import re
import pandas as pd
import jams

from jams.util import find_with_extension

__curator__ = dict(name='Rachel Bittner', email='rachel.bittner@nyu.edu')
__corpus__ = 'MedleyDB'


def medleydb_file_metadata(infile):
    '''Construct a metadata object from an SMC wav file'''

    match = re.match('.*/(?P<artist>.*?)_(?P<title>.*)_MIX.wav$', infile)

    if not match:
        raise RuntimeError('Could not index filename {:s}'.format(infile))

    # Get the duration of the track
    y, sr = librosa.load(infile, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    # Format duration as time
    metadata = jams.FileMetadata(title=match.group('title'),
                                 artist=match.group('artist'),
                                 duration=duration)

    return metadata


def get_output_file(output_dir, metadata):
    outfile = os.extsep.join(['_'.join([metadata.artist,
                                        metadata.title]),
                             'jams'])

    return os.path.join(output_dir, outfile)


def save_jam(output_dir, jam):
    '''Save the output jam'''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    outfile = get_output_file(output_dir, jam.file_metadata)

    print 'Saving {:s}'.format(outfile)
    jam.save(outfile)


def instrument_source(input_dir, wav):
    '''Get the instrument source activations for a particular file'''

    root = jams.util.filebase(wav).replace('_MIX', '')

    source_lab = os.path.join(input_dir,
                              'Annotations',
                              'Instrument_Activations',
                              'SOURCEID',
                              root + '_SOURCEID.lab')

    # Load the csv into a dict
    data = pd.read_csv(source_lab, sep=',')
    data['duration'] = data['end_time'] - data['start_time']
    del data['end_time']
    data['time'] = data['start_time']
    del data['start_time']

    data['value'] = data['instrument_label']
    del data['instrument_label']
    data['confidence'] = 1

    data = data.sort(columns=['time', 'duration', 'value'])
    data = jams.JamsFrame.from_dataframe(data)

    # Make an annotation
    curator = jams.Curator(**__curator__)
    metadata = jams.AnnotationMetadata(curator=curator, corpus=__corpus__)

    annotation = jams.Annotation('tag_medleydb_instruments',
                                 data=data,
                                 annotation_metadata=metadata)

    return annotation


def parse_medleydb(input_dir=None, output_dir=None, skip=False):
    '''Convert medleydb to jams'''

    # Get a list of the wavs
    wav_files = find_with_extension(os.path.join(input_dir,
                                                 'Audio'),
                                    'wav', depth=2)

    for wav in wav_files:
        # Get the file metadata
        metadata = medleydb_file_metadata(wav)

        if skip and os.path.exists(get_output_file(output_dir, metadata)):
            continue

        jam = jams.JAMS(file_metadata=metadata)

        # Get the instrument source activations
        jam.annotations.append(instrument_source(input_dir, wav))

        # Add content path to the top-level sandbox
        jam.sandbox.content_path = os.path.basename(wav)

        # Save the jam
        save_jam(output_dir, jam)


def parse_arguments(args):

    parser = argparse.ArgumentParser(description='MedleyDB annotation parser')

    parser.add_argument('input_dir',
                        type=str,
                        help='Path to the MedleyDB root directory')

    parser.add_argument('output_dir',
                        type=str,
                        help='Path to output jam files')

    parser.add_argument('-s', '--skip', dest='skip', action='store_true',
                        help='Skip files that have already been computed')

    return vars(parser.parse_args(args))


if __name__ == '__main__':
    parameters = parse_arguments(sys.argv[1:])

    parse_medleydb(**parameters)
