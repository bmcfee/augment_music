#!/usr/bin/env python
'''Chef Boyardeep: canned pasta'''

import six

import pandas as pd
import numpy as np

from lasagne.layers import get_all_params
import theano

from sklearn.base import ClassifierMixin, BaseEstimator, TransformerMixin


class Boyardeep(BaseEstimator, ClassifierMixin, TransformerMixin):

    def __init__(self, architecture, multilabel=False, regression=False,
                 callback=None):
        '''Initialize a chef boyardeep model.

        Parameters
        ----------
        architecture : dict
            the model architecture.

            `architecture['layers']` contains tuples
            `(layer_type, params_dict)`
            where `layer_type` is a Lasagne layer class,
            and `params_dict` contains the parameters of the layer
            as `params_dict['args']` and `params_dict['kwargs']`.

            `architecture['update']` contains a single tuple
            `(update_function, params_dict)`
            where `update_function` is one of `lasagne.updates.*`
            and `params_dict` is as before.

            `architecture['cost']` contains a single function, eg,
            `theano.nnet.binary_entropy` to by applied against
            the target and final layer output.
            The actual score will be the mean of the outputs over the batch.

            `architecture['output']` is a Theano class specifying the output
            type, eg, `theano.tensor.ivector`


        callback : None or callable
            An optional function to call after each iteration
        '''

        if callback is None or six.callable(callback):
            self.callback = callback
        else:
            raise TypeError('callback must be None or callable')

        self.multilabel = multilabel
        self.regression = regression
        self._construct_model(architecture)

    def _construct_model(self, arch):
        '''Construct the model'''

        # Lay down the model
        layers = None

        for layer_type, layer_params in arch['layers']:
            if layers is None:
                layers = [layer_type(*layer_params.get('args', []),
                                     **layer_params.get('kwargs', {}))]
            else:
                layers.append(layer_type(layers[-1],
                                         *layer_params.get('args', []),
                                         **layer_params.get('kwargs', {})))

        # Bake the functions
        target = arch['output'](name='target')

        cost = arch['cost'](layers[-1].get_output(), target).mean()

        # Compute updates, there's also SGD with momentum, Adagrad, etc.
        updates = arch['update'][0](cost,
                                    get_all_params(layers[-1]),
                                    *arch['update'][1].get('args', []),
                                    **arch['update'][1].get('kwargs', {}))

        self.layers = layers

        # Compile theano functions for train/test
        self.train = theano.function([layers[0].input_var, target], cost,
                                     updates=updates)

        # Other useful functions
        self.cost = theano.function([layers[0].input_var, target], cost)

        self.output = theano.function([layers[0].input_var],
                                      layers[-1].get_output())

        self.output_d = theano.function([layers[0].input_var],
                                        layers[-1].get_output(deterministic=True))

        if len(layers) > 1:
            self.feature = theano.function([layers[0].input_var],
                                           layers[-2].get_output(deterministic=True))
        else:
            self.feature = theano.function([layers[0].input_var],
                                           layers[0].input_var)

        self.n_ = 0
        self.n_batches_ = 0
        self.train_cost_ = pd.Series(name='cost_train')

        self._arch = arch

    def partial_fit(self, X, y):
        '''Do a partial update'''

        # Take a step
        self.train(X, y)

        # Increment the counters
        self.n_batches_ += 1
        self.n_ += len(X)

        # Cache the cost on this batch
        self.train_cost_.set_value(self.n_batches_, self.cost(X, y))

        # Hit the callback
        if six.callable(self.callback):
            self.callback(self)

    fit = partial_fit

    def predict(self, X):

        if self.multilabel:
            return (self.output_d(X) >= 0.5).astype(np.int32)
        elif self.regression:
            return self.output_d(X)
        else:
            return np.argmax(self.output_d(X), axis=1)

    def predict_proba(self, X):

        return self.output_d(X)

    def transform(self, X):

        return self.feature(X)
