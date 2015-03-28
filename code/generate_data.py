#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Generate augmented data from jams'''

from __future__ import print_function

from argparse import ArgumentParser
from glob import glob

import os
import sys

import muda
import jams

from joblib import Parallel, delayed


def make_pipeline(plfile):
    '''Construct the audio deformation pipeline'''

    with open(plfile, 'r') as f:
        pljson = '\n'.join([line for line in f])
        return muda.deserialize(pljson)


def audio_path(jam, content_path):
    '''Find the audio from a jam'''

    return os.path.join(content_path, jam.sandbox.content_path)


def augment_data(transformer, i, ann_file, content_path, output_dir, fmt):
    '''Perform data augmentation'''

    # First, load the jam
    jam = jams.load(ann_file)

    # Get the audio path
    audio = audio_path(jam, content_path)

    # Load the audio
    jam = muda.load_jam_audio(jam, audio)

    for j, jam_aug in enumerate(transformer.transform(jam)):
        out_base = os.path.join(output_dir, '{:05d}_{:05d}'.format(i, j))
        out_audio = os.path.extsep.join([out_base, fmt])
        out_jam = os.path.extsep.join([out_base, 'jams'])
        muda.save(out_audio, out_jam, jam_aug,
                  exclusive_creation=False)
    print('Finished {:05d} | {:s}'.format(i, os.path.basename(ann_file)))


def run(pipeline=None, annotation_path=None, content_path=None,
        output_dir=None, num_jobs=2, fmt='ogg'):
    '''Do the needful'''

    # Step 1: load in the transformation
    transformer = make_pipeline(pipeline)

    # Step 2: absorb the files
    anns = sorted(glob(os.path.join(annotation_path, '*.jams')))

    # Step 3: schedule the jobs
    AD = delayed(augment_data)
    Parallel(n_jobs=num_jobs, verbose=1)(AD(transformer, i, ann,
                                            content_path, output_dir, fmt)
                                         for (i, ann) in enumerate(anns))


def get_params(args):
    '''Process command-line arguments'''

    parser = ArgumentParser(description='Execute a muda pipeline')

    parser.add_argument('pipeline', type=str,
                        help='Path to the pipeline json')
    parser.add_argument('annotation_path', type=str,
                        help='Path to the annotations')
    parser.add_argument('content_path', type=str,
                        help='Path to the audio content')
    parser.add_argument('output_dir', type=str,
                        help='Path to store the output files')

    parser.add_argument('-n', '--num_jobs', dest='num_jobs',
                        type=int, default=2, help='Number of parallel threads')

    parser.add_argument('-f', '--format', dest='fmt', type=str, default='ogg',
                        help='Output audio format')

    return vars(parser.parse_args(args))


if __name__ == '__main__':
    params = get_params(sys.argv[1:])

    run(**params)
