#!/usr/bin/env python
'''Boyardeep canning process'''

import jsonpickle

jsonpickle.set_encoder_options('simplejson', indent=2)


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

    with open(jsfile, mode='r') as fdesc:
        plj = ''.join([_ for _ in fdesc])
        return jsonpickle.decode(plj, **kwargs)


def save_architecture(jsfile, arch, **kwargs):
    '''Save an architecture specification to a json file

    Parameters
    ----------
    jsfile : path

    arch : dict
        See `Boyardeep.__init__`

    kwargs
        Additional parameters to jsonpickle.encode
    '''

    with open(jsfile, mode='w') as fdesc:
        fdesc.write(jsonpickle.encode(arch, **kwargs))
