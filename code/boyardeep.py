#!/usr/bin/env python
'''Chef Boyardeep!  He cooks up lasagne.'''

import six
import jsonpickle

import numpy as np
import scipy

import lasagne
from lasagne.layers import get_all_params
import theano
import theano.tensor as T

from sklearn.base import ClassifierMixin, BaseEstimator


class Boyardeep(BaseEstimator, ClassifierMixin):

    def __init__(self, architecture, learning_rate=1e-2, momentum=0.9,
                 callback=None):
        '''Initialize a chef boyardeep model.

        Parameters
        ----------
        architecture : list of tuples
            the model architecture, described as tuples
            `(layer_type, params_dict)`
            where `layer_type` is a Lasagne layer class,
            and `params_dict` contains the parameters of the layer
            as `params_dict['args']` and `params_dict['kwargs']`.

        learning_rate : float
        momentum : float
            Update parameters

        callback : None or callable
            An optional function to call after each iteration
        '''

        if callback is None or six.callable(callback):
            self.callback = callback
        else:
            raise TypeError('callback must be None or callable')

        self.architecture = architecture

        self.learning_rate = learning_rate
        self.momentum = momentum

        self._construct_model()

    def _construct_model(self):
        '''Construct the model'''

        # Lay down the model
        layers = None

        for layer_type, layer_params in self.architecture:
            if layers is None:
                layers = [layer_type(*layer_params['args'],
                                     **layer_params['kwargs'])]
            else:
                layers.append(layer_type(layers[-1],
                                         *layer_params['args'],
                                         **layer_params['kwargs']))

        # Bake the functions
        target = T.matrix(name='target')

        cost = T.nnet.binary_crossentropy(layers[-1].get_output(),
                                          target).mean()

        # Compute updates, there's also SGD with momentum, Adagrad, etc.
        updates = lasagne.updates.rmsprop(cost,
                                          get_all_params(layers[-1]),
                                          self.learning_rate,
                                          self.momentum)

        # Compile theano functions for train/test
        train = theano.function([layers[0].input_var, target],
                                cost,
                                updates=updates)

        # Other useful functions
        compute_cost = theano.function([layers[0].input_var, target],
                                       cost)

        output = theano.function([layers[0].input_var],
                                 layers[-1].get_output())

        self.layers = layers
        self.train = train
        self.compute_cost = compute_cost
        self.output = output

    def partial_fit(self, X, y):
        '''Do a partial update'''

        self.train(X, y)

        self.callback(self)

    fit = partial_fit


def load_architecture(jsfile, **kwargs):
    '''Load an architecture specification from a json file

    Parameters
    ----------
    jsfile : path or readable
        Where the architecture file lives

    kwargs
        Additional parameters to jsonpickle.decode

    Returns
    -------
    architecture : list of tuples
        Deserialized architecture
    '''

    with open(jsfile, mode='r') as fd:
        plj = ''.join([_ for _ in fd])
        return jsonpickle.decode(plj, **kwargs)
