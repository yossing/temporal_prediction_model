import sys
import os
from imp import reload
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from analysis import quantify_aud_strfs as quant
from analysis import auditory_analysis as aan
from visualisation import network_visualisation as vis
reload(aan)
reload(vis)
reload(quant)

def load_pickled_data(fp):
    import pickle
    import gzip
    import numpy

    with open(fp, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()

def postprocess_res_folder(res_folder_path, save_path, real_path='/path/to/real/strfs/.pkl'):
    res_pd = compile_sparse_results_pd(res_folder_path)
    rstrfs = pkl.load(open(real_path,'rb'))
    res_pd = compare_real_model_populations(res_pd, rstrfs)
    res_pd = quant.add_mean_ks(res_pd)
    res_pd.to_pickle(save_path)
    return 
def compile_sparse_results_pd(res_folder_path):
    res_pd = pd.DataFrame

    #Loop through all files in folder and add to compiled results
    filenames = os.listdir(res_folder_path)
    ix = 0
    for fn in filenames:
        res_folder = os.path.join(res_folder_path, fn)
        if os.path.isdir(res_folder):
            this_respath = os.path.join(res_folder, "python_network.pkl")
            if os.path.isfile(this_respath):
                this_res = load_pickled_data(this_respath)
                this_res_pd= pd.DataFrame(this_res['network_settings'], index = [ix])
                this_res_pd['final_train_cost'] = this_res['cost_history']['train_costs'][-1]
                if 'val_costs' in this_res['cost_history'].keys():
                    this_res_pd['final_val_cost'] = this_res['cost_history']['val_sse'][-1]
                this_res_pd['n_epochs_run'] = len(this_res['cost_history']['train_costs'])
                this_res_pd['results_path'] = this_respath
                this_res_pd['log_lambdav'] = np.log10(this_res_pd['lambdav']).round(2)
                if res_pd.empty:
                    res_pd = this_res_pd
                else:
                    res_pd = res_pd.append(this_res_pd)
                ix+=1
    res_pd.to_pickle(os.path.join(res_folder_path, "results_pd.pkl"))
    return res_pd
def compare_real_model_populations(this_pd, rstrfs, display=1, keep_prop=0.01):

    n_h = 40
    num_freq = 32
    out_pd = this_pd.copy()
    ii= 0
    for entry_loc,this_entry in this_pd.iterrows():

        pth = this_entry['results_path']

        Phi = load_pickled_data(pth)['Phi']
        weights = Phi.T
        l2_norm = np.sum(weights**2,axis=1)
        keep_ix = l2_norm>keep_prop*max(l2_norm)
        mstrfs = weights[keep_ix,:]
        num_mstrfs = mstrfs.shape[0]
        mstrfs =np.reshape(mstrfs,[num_mstrfs, n_h, num_freq])
        mstrfs = np.rollaxis(mstrfs,2,1)

        if ii == 0:
            temp = quant.compare_real_model_distributions(mstrfs, rstrfs, this_entry)
            for cname in temp.index:
                if cname not in out_pd.columns:
                    temp[cname] = None
                else:
                    temp = temp.drop(cname)
            out_pd = out_pd.assign(**temp)
        
        this_entry = quant.compare_real_model_distributions(mstrfs, rstrfs, this_entry)

        out_pd.loc[entry_loc,this_entry.index.tolist()] = this_entry

        ii+=1
    return pd.DataFrame(out_pd)


def get_weights_from_pd_index(this_pd, ix, keep_prop=0.1, n_h=40, n_f=32):
    pth = this_pd.iloc[ix].results_path
    print(pth)
    weights = get_weights_from_path(pth, keep_prop=keep_prop, n_h = n_h, n_f = n_f)
    return weights

def get_weights_from_path(pth, keep_prop=0.1, n_h=40, n_f=32):
    Phi = load_pickled_data(pth)['Phi']
    weights = Phi.T
    l2_norm = np.sum(weights**2,axis=1)
    keep_ix = l2_norm>keep_prop*max(l2_norm)
    mstrfs = weights[keep_ix,:]
    num_mstrfs = mstrfs.shape[0]
    mstrfs =np.reshape(mstrfs,[num_mstrfs, n_h, n_f])
    mstrfs = np.rollaxis(mstrfs,2,1)

    # mstrfs = mstrfs[:,:,2:]
    print(mstrfs.shape)
    return mstrfs

def plot_these_sparse_weights(weights, n_h=40, num_freq=32, keep_prop = 0.01, figsize=(12,12), ax=None):  
    l2_norm = np.sum(weights**2,axis=0)
    keep_ix = l2_norm>keep_prop*max(l2_norm)
    weights = weights[:,keep_ix] 

    l2_norm = l2_norm[keep_ix]
    sort_ix = np.argsort(l2_norm);

    weights = weights[:,sort_ix] 

    weights = weights.transpose()
    numweights = weights.shape[0]
    weights =np.reshape(weights,[numweights, n_h, num_freq])
    weights = np.rollaxis(weights,2,1)

    [weight_image, weight_image_norm] = vis.getWeightImage(weights,num_freq,n_h)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    ax.set_xticks(np.arange(0,weight_image_norm.shape[1], n_h))
    ax.set_yticks(np.arange(0,weight_image_norm.shape[0], num_freq))
    im = ax.imshow(weight_image_norm, cmap='seismic')
    plt.grid()
    # plt.show()
    return [ax,im]
