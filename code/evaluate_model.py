#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Evaluation script for optimus models'''

from __future__ import print_function

import argparse
import json
import sys
import os
from itertools import product

import numpy as np
import pandas as pd
import sklearn.metrics as skm
import sklearn.preprocessing as skp
from joblib import Parallel, delayed

import optimus

import data_generator as dg

from train_optimus_model import INSTRUMENTS


def process_arguments(args):
    '''Parse arguments'''

    parser = argparse.ArgumentParser(description='Evaluate instrument model')

    parser.add_argument('-i', '--input-path', dest='input_path',
                        type=str,
                        required=True,
                        help='Path to the input data')

    parser.add_argument('-t', '--training-index', dest='index_file',
                        type=str, required=True,
                        help='Path to the augmentation index')

    parser.add_argument('-s', '--split', dest='split_file',
                        type=str, required=True,
                        help='Path to the train/test split file')

    parser.add_argument('-n', '--split-name', dest='split_name',
                        type=str, required=False,
                        default='test',
                        help='Name of the split to validate. Default: "test"')

    parser.add_argument('-j', '--num-jobs', dest='num_jobs',
                        type=int, required=False,
                        default=1,
                        help='Number of jobs to run in parallel')

    parser.add_argument('-m', '--model-file', dest='model_file',
                        type=str, required=True,
                        help='Path to the model specification json')

    parser.add_argument('-p', '--parameters', dest='model_parameters',
                        type=str, required=True,
                        help='Path to the model parameters npz')

    parser.add_argument('-d', '--output-dir', dest='output_path',
                        type=str, required=True,
                        help='Path to store the predictions')

    parser.add_argument('--predict', dest='predict',
                        action='store_true', default=False,
                        help='Store predictions')

    parser.add_argument('-o', '--output', dest='score_file',
                        type=str, required=True,
                        help='Path to store the results as json')

    return parser.parse_args(args)


def get_predictor_input_size(predictor):
    port = predictor.ports['layer0.input']
    return port.shape[2]


def evaluator(predictor, LT, val_id, aug_id, predict, input_path,
              output_directory):

    key = dg.augment_file_id(val_id, aug_id)

    results = {}

    # Get the patch size from the predictor
    n_columns = get_predictor_input_size(predictor)
    samples = dg.stream_data(key, input_path, LT, n_columns=n_columns)

    y_true = []
    y_score = []

    for data in samples:
        y_true.append(data['Y'])
        scores = predictor(data['X'])['Z']
        y_score.append(scores)

    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    y_pred = (y_score >= 0.5).astype(int)

    #   save the predictions, truth, and scores out to an npz file
    if predict:
        outfile = os.path.join(output_directory, os.extsep.join([key, 'npz']))
        np.savez(outfile, {'y_true': y_true,
                           'y_score': y_score,
                           'y_pred': y_pred,
                           'classes': LT.classes_})

    results['lrap'] = skm.label_ranking_average_precision_score(y_true,
                                                                y_score)

    results['hamming'] = skm.hamming_loss(y_true,
                                          y_pred)

    results['f1macro'] = skm.f1_score(y_true,
                                      y_pred,
                                      average='macro')

    results['f1micro'] = skm.f1_score(y_true,
                                      y_pred,
                                      average='micro')

    results['f1samples'] = skm.f1_score(y_true,
                                        y_pred,
                                        average='samples')

    results['accuracy'] = skm.accuracy_score(y_true, y_pred)

    results['support'] = len(y_true)

    return {key: results}


def main(args):
    '''Main function'''

    # Load the split file
    with open(args.split_file, 'r') as fdesc:
        split_file = json.load(fdesc)

    # Slice out the validation ids
    validation_ids = split_file[args.split_name]

    # Load the augmentation index
    aug_ids = np.atleast_1d(np.loadtxt(args.index_file,
                                       dtype=int))

    # Load the model
    predictor = optimus.load(args.model_file, args.model_parameters)

    # Make the list of evaluation points
    probes = product(validation_ids, aug_ids)

    # Make the label encoder
    label_encoder = skp.MultiLabelBinarizer(classes=INSTRUMENTS)
    label_encoder.fit(INSTRUMENTS)

    # Fan out the results
    d_eval = delayed(evaluator)
    results = {}
    for res in Parallel(n_jobs=args.num_jobs)(d_eval(predictor,
                                                     label_encoder,
                                                     val_id, aug_id,
                                                     args.predict,
                                                     args.input_path,
                                                     args.output_path)
                                              for (val_id, aug_id) in probes):
        results.update(res)

    rframe = pd.DataFrame.from_dict(results, orient='index')

    print(rframe.describe())
    rframe.to_json(os.path.join(args.output_path, args.score_file))


if __name__ == '__main__':

    parameters = process_arguments(sys.argv[1:])

    pd.set_option('display.precision', 4)
    main(parameters)
