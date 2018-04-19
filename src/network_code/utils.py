from scipy.ndimage.interpolation import shift
import numpy as np
import itertools
import os
import ntpath
import pickle as pkl
	
def explode_to_list(d):
    '''
    Convert a dictionary of lists into a list of dictionaries of every possible combination
    '''
    exploded_list = []
    try:
        for vals in itertools.product(*d.values()):
            exploded_list.append(dict(zip(d.keys(), vals)))
    #This will fail when all of the dict entries have only one element. 
    #Handle this case with a catch:
    except TypeError:
        exploded_list.append(d)
    return exploded_list

from time import time
_tstart_stack = []

def tic():
    _tstart_stack.append(time())

def toc(display=True, return_t=False):
    t_elapsed = time() - _tstart_stack.pop()
    if display:
        fmt = "Elapsed: %2f s"
        print(fmt % (t_elapsed))
    if return_t:
        return t_elapsed

def unpickle_data(load_path):
    load_path = os.path.expanduser(load_path)
    with open(load_path, "rb") as f:
        dat = pkl.load(f)
    return dat

def pickle_data(data, save_path, protocol=4, create_par_dirs=True):
    save_path = os.path.expanduser(save_path)
    if not os.path.exists(os.path.dirname(os.path.abspath(save_path))) and create_par_dirs:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)))
    with open(save_path, "wb") as f:
        pkl.dump(data, f, protocol=protocol)
    return


def convert_nans(d):
    # d_out = OrderedDict()
    d_out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            d_out[k] = convert_nans(v)
        else:
            if type(v) == float and np.isnan(v):
                print('converting nan to None for key:', k)
                v = None
            d_out[k] = v
    return d_out