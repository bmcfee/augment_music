#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Generate augmented data from jams'''

from __future__ import print_function

from argparse import ArgumentParser

import os
import sys

import muda
import jams

from joblib import Parallel, delayed


def make_pipeline(plfile):
    '''Construct the audio deformation pipeline'''

    with open(plfile, 'r') as fdesc:
        pljson = '\n'.join([line for line in fdesc])
        return muda.deserialize(pljson)


def audio_path(jam, content_path):
    '''Find the audio from a jam'''

    return os.path.join(content_path, jam.sandbox.content_path)


def augment_data(transformer, ann_file, content_path, output_dir, fmt):
    '''Perform data augmentation'''

    # First, load the jam
    jam = jams.load(ann_file)

    # Get the audio path
    audio = audio_path(jam, content_path)

    # Load the audio
    jam = muda.load_jam_audio(jam, audio)

    root_name = jams.util.filebase(ann_file)

    for j, jam_aug in enumerate(transformer.transform(jam)):
        out_base = os.path.join(output_dir, '{}_{:05d}'.format(root_name, j))
        out_audio = os.path.extsep.join([out_base, fmt])
        out_jam = os.path.extsep.join([out_base, 'jams'])
        muda.save(out_audio, out_jam, jam_aug,
                  exclusive_creation=False)
    print('Finished {}'.format(root_name))


def run(pipeline=None, jam_files=None, content_path=None,
        verbose=0, output_dir=None, num_jobs=2, fmt='ogg'):
    '''Do the needful'''

    # Step 1: load in the transformation
    transformer = make_pipeline(pipeline)

    # Step 3: schedule the jobs
    augment = delayed(augment_data)

    Parallel(n_jobs=num_jobs,
             verbose=verbose)(augment(transformer, ann,
                                      content_path,
                                      output_dir, fmt)
                              for ann in jam_files)


def get_params(args):
    '''Process command-line arguments'''

    parser = ArgumentParser(description='Execute a muda pipeline')

    parser.add_argument('-n', '--num_jobs', dest='num_jobs',
                        type=int, default=1, help='Number of parallel threads')

    parser.add_argument('-f', '--format', dest='fmt', type=str, default='ogg',
                        help='Output audio format')

    parser.add_argument('-v', '--verbose', type=int, default=0,
                        help='Verbosity')

    parser.add_argument('pipeline', type=str,
                        help='Path to the pipeline json')

    parser.add_argument('content_path', type=str,
                        help='Path to the audio content')

    parser.add_argument('output_dir', type=str,
                        help='Path to store the output files')

    parser.add_argument('jam_files', type=str, nargs='+',
                        help='Path to the annotations')

    return vars(parser.parse_args(args))


if __name__ == '__main__':
    params = get_params(sys.argv[1:])

    run(**params)
