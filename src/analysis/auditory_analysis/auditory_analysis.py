import os    
import importlib
from imp import reload
import numpy as np
import pandas as pd
import pickle as pkl
import quantify_aud_strfs as quant
import results_dataframe_handling as rdh
reload(rdh)
reload(quant)

def postprocess_res_folder(res_folder_path, save_path, real_path):
    res_pd = rdh.compile_results_pd(res_folder_path)
    rstrfs = pkl.load(open((real_path, 'rb')))
    res_pd = quant.compare_real_model_populations(res_pd, rstrfs)
    res_pd = quant.add_mean_ks(res_pd)
    res_pd.to_pickle(save_path)


def perform_mds(list_of_ws, n_examples=100, n_components=2):
    w_2d = []
    for w in list_of_ws:
        #shuffle along first dimension, so get random selection of examples from full population
        np.random.shuffle(w)
        #select n_examples from this set of weights
        w = w[:n_examples,...]
        #collapse last dimensions        
        w = np.reshape(w, [w.shape[0],-1])
        #normalise each example to lie between -1 and 1
        for ix in range(n_examples):
            this_w = w[ix,:]
            if np.max(np.abs(this_w))>0:
                w[ix,:]=this_w/np.max(np.abs(this_w))
        w_2d.append(w)
    X = np.concatenate(w_2d, axis=0)

    mds = manifold.MDS(n_components, max_iter=100, n_init=1)
    Y = mds.fit_transform(X)
    return Y  
