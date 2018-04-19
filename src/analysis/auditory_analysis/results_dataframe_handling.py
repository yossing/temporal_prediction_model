import pandas as pd
import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import traceback


def compile_results_pd(res_folder_path, include_weights=False):
    res_pd = pd.DataFrame

    #Loop through all files in folder and add to compiled results
    filenames = os.listdir(res_folder_path)
    ix = 0
    for fn in filenames:
        res_folder = os.path.join(res_folder_path, fn)
        if os.path.isdir(res_folder):
            this_respath = os.path.join(res_folder, "python_network.pkl")
            if os.path.isfile(this_respath):
                try:
                    this_res = pkl.load(open(this_respath, 'rb'))
                    this_res_pd = pd.DataFrame(this_res.network_settings, index = [ix])
                    if 'conv_settings' in this_res.__dict__:
                        temp_pd = pd.DataFrame(this_res.conv_settings, index = [ix])
                        this_res_pd = pd.concat([this_res_pd,temp_pd], axis = 1, join='inner')
                    temp_pd = pd.DataFrame(this_res.cost_settings, index = [ix])
                    this_res_pd = pd.concat([this_res_pd,temp_pd], axis = 1, join='inner')
                    this_res_pd['log_reg_factor'] = np.log10(this_res_pd['reg_factor']).round(2)
                    this_res_pd['final_val_cost'] = this_res.cost_history['final_val_cost']
                    this_res_pd['final_train_cost'] = this_res.cost_history['final_train_cost']
                    this_res_pd['n_epochs_run'] = len(this_res.cost_history['val_costs'])
                    this_res_pd['results_path'] = this_respath

                    if 'input_noise_ratio' in this_res.input_settings:
                        this_res_pd['input_noise_ratio'] = this_res.input_settings['input_noise_ratio']
                    if 'noise_ratio' in this_res.input_settings:
                        this_res_pd['noise_ratio'] = this_res.input_settings['noise_ratio']

                    this_res_pd = this_res_pd.loc[:,~this_res_pd.columns.duplicated()]                    
    #                 this_res_pd['pn'] = this_res
                    if include_weights:
                        if 'network_params' in this_res.__dict__:
                            network_param_values = this_res.network_params
                        else:
                            network_param_values = this_res.network_param_values
                        this_res_pd['network_param_values'] = [network_param_values]
                        

                    if res_pd.empty:
                        res_pd = this_res_pd
                    else:
                        res_pd = res_pd.append(this_res_pd)
                    ix+=1

                except Exception as e:
                    print('could not load %s' %this_respath)
                    traceback.print_exc()
                    # print(e)

    if 'act_reg_factor' in res_pd.keys():
        res_pd.loc[res_pd['act_reg_factor'].isnull(),'act_reg_factor'] = 0
    return res_pd
    if 'reg_factor' in res_pd.keys():
        res_pd.loc[res_pd['reg_factor'].isnull(),'reg_factor'] = 0
    return res_pd
# res_pd = pd.read_pickle(os.path.join(res_folder_path, "results_pd.pkl"))

def plot_3D_info_grid(res_pd, key1, key2, key3, ax=None,  ax_labels=None):
    mx_key3_val = res_pd.sort_values(key3).iloc[-1][key3]
#     print(mx_val_cost)
    sorted_pd = res_pd.sort_values(key1)
    key2_sorted_pd = res_pd.sort_values(key2)

    cost_grid_im = np.empty([len(sorted_pd[key1].unique()), len(sorted_pd[key2].unique())])
    
    for ii, k1 in enumerate(sorted_pd[key1].unique()):
        temp = np.ones([len(sorted_pd[key2].unique())])*mx_key3_val*1
        temp_pd = sorted_pd[sorted_pd[key1]==k1].sort_values(key2)
        for jj, k2 in enumerate(key2_sorted_pd[key2].unique()):
            if k2 in temp_pd[key2].unique():              
                temp[jj] = temp_pd[temp_pd[key2]==k2][key3]
        cost_grid_im[ii,:] = temp
    y_tick_labels = sorted_pd[key1].unique()
    x_tick_labels = sorted_pd.sort_values(key2)[key2].unique()
    
    #plotting code
    gs = plt.GridSpec(8,10)
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(gs[:,:9])

    im = ax.imshow(cost_grid_im, cmap='jet', interpolation='nearest')
    ax.set_yticks(np.arange(len(y_tick_labels)))
    ax.set_yticklabels(y_tick_labels)
    ax.set_xticks(np.arange(len(x_tick_labels)))
    ax.set_xticklabels(x_tick_labels)
    if ax_labels is None:
        ax.set_ylabel(key1, fontsize='x-large')
        ax.set_xlabel(key2, fontsize='x-large')
    else:
        ax.set_xlabel(ax_labels[0], fontsize='x-large')
        ax.set_ylabel(ax_labels[1], fontsize='x-large')    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.35)
    plt.colorbar(im,cax=cax)
    if ax_labels is None: 
        cax.set_ylabel(key3, fontsize='x-large')
    else:
        cax.set_ylabel(ax_labels[2], fontsize='x-large')
    # return [ax,cax]
    return cost_grid_im
