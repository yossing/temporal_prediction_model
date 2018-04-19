import os
import importlib
import random
from imp import reload
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import matplotlib.pyplot as plt
import visualisation as vis
import auditory_analysis as aan
import quantify_aud_strfs as quant
import sparse_analysis as san
import plotting_funcs as plotting_funcs
reload(plotting_funcs)
reload(san)
reload(quant)
reload(aan)
reload(vis)

def plot_quant_aud_figure(weights, sparse_weights, rstrfs, 
                          Y_mds, t_pow_real, t_pow_model, 
                          t_pow_sparse, save_path=None, use_blobs=False):

    fig = plt.figure(figsize=[10,20])
    gs = plt.GridSpec(200,100)
    pop_ax = np.empty((2,3), dtype=object)
    hist_ax = np.empty((2,2), dtype=object)

    mds_ax = fig.add_subplot(gs[:45,:45])
    temp_ax = fig.add_subplot(gs[:45,64:])

    pop_ax[0,0] = fig.add_subplot(gs[55:85,:30])
    pop_ax[0,1] = fig.add_subplot(gs[55:85, 35:65])
    pop_ax[0,2] = fig.add_subplot(gs[55:85, 70:])

    hist_ax[0,0] = fig.add_subplot(gs[95:110, :45])
    hist_ax[0,1] = fig.add_subplot(gs[95:110, 55:])

    pop_ax[1,0] = fig.add_subplot(gs[120:150,:30])
    pop_ax[1,1] = fig.add_subplot(gs[120:150, 35:65])
    pop_ax[1,2] = fig.add_subplot(gs[120:150, 70:])

    hist_ax[1,0] = fig.add_subplot(gs[160:175, :45])
    hist_ax[1,1] = fig.add_subplot(gs[160:175, 55:])

    # gs.update(wspace=0.005, hspace=0.05) # set the spacing between axes. 
    labels = ['A1', 'predict', 'sparse', 'noise']
    azure = '#006FFF'
    clors = ['red','black',azure,'y']# markers = ['o', '_', '|']
    markers = ['8', 'o', '^']

    # mds_ax.set_axis_off()
    # mds_ax.margins = [0,0]
    mds_axlim = 20
    mds_axes = plotting_funcs.plot_2D_projection_with_hists(Y_mds,mds_ax, 
                                                            labels =None, 
                                                            clors = clors, 
                                                            markers = markers)
    mds_axes[0].set_xlim([-mds_axlim,mds_axlim])
    mds_axes[0].set_ylim([-mds_axlim,mds_axlim])

    for ax in mds_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()

    ax = temp_ax
    divider = make_axes_locatable(ax)
    aspect = 4
    pad_fraction = 0.05
    width = axes_size.AxesY(ax, aspect=1/aspect)
    pad = axes_size.Fraction(pad_fraction, width)
    dummy_ax_x = divider.append_axes("top", size=width, pad=0.1, sharex=ax)
    # dummy_ax_y = divider.append_axes("right", size=width, pad=0.1, sharey=ax)
    dummy_ax_x.set_axis_off()
    # dummy_ax_y.set_axis_off()
    if use_blobs:
        markers = ['o', 'o', 'o']
    else:
        markers = ['blob', 'blob', 'blob']

    t_plot_kwargs = {}
    t_plot_kwargs['color']=clors[0]
    t_plot_kwargs['label']='A1'
    t_plot_kwargs['linewidth']=3
    # ax.plot(real_measures.t_pow.mean(axis=0)/sum(real_measures.t_pow.mean(axis=0)),'o', **t_plot_kwargs )
    # t_plot_kwargs['label']=''
    ax.plot(t_pow_real, **t_plot_kwargs)

    t_plot_kwargs = {}
    t_plot_kwargs['color']=clors[1]
    t_plot_kwargs['label']='prediction'
    t_plot_kwargs['linewidth']=3

    # ax.plot(res_pd.iloc[0].t_pow.mean(axis=0)[2:]/sum(res_pd.iloc[0].t_pow.mean(axis=0)[2:]),'_', **t_plot_kwargs)
    # t_plot_kwargs['label']=''
    ax.plot(t_pow_model, **t_plot_kwargs)


    t_plot_kwargs['color']=clors[2]
    t_plot_kwargs['label']='sparse'
    t_plot_kwargs['linewidth']=3

    # ax.plot(sparse_pd.iloc[11].t_pow.mean(axis=0)[2:]/sum(sparse_pd.iloc[11].t_pow.mean(axis=0)[2:]),'|', **t_plot_kwargs)
    # t_plot_kwargs['label']=''
    ax.plot(t_pow_sparse, **t_plot_kwargs)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    temp_ax = ax
    temp_ax.set_xticklabels(['-250', '-200','-150','-100','-50', '0'])
    temp_ax.set_ylabel('Proportion of power')
    temp_ax.set_xlabel('Time (msec)')



    [mf_peak_pos, mf_peak_neg, mf_bw_pos, mf_bw_neg, mt_peak_pos, mt_peak_neg, mt_bw_pos, mt_bw_neg, m_pow] = quant.quantify_strfs(weights)
    [sf_peak_pos, sf_peak_neg, sf_bw_pos, sf_bw_neg, st_peak_pos, st_peak_neg, st_bw_pos, st_bw_neg, s_pow] = quant.quantify_strfs(sparse_weights)
    [rf_peak_pos, rf_peak_neg, rf_bw_pos, rf_bw_neg, rt_peak_pos, rt_peak_neg, rt_bw_pos, rt_bw_neg, r_pow] = quant.quantify_strfs(rstrfs, n_h=38)

    mf_ix = [mf_bw_neg>0] #and [mf_bw_pos>0]
    rf_ix = [rf_bw_neg>0] #and [rf_bw_pos>0]
    mt_ix = [mt_bw_neg>0] #and [mt_bw_pos>0]
    rt_ix = [rt_bw_neg>0] #and [rt_bw_pos>0]
    sf_ix = [sf_bw_neg>0] #and [mf_bw_pos>0]
    st_ix = [st_bw_neg>0] #and [mt_bw_pos>0]

    print('Excluding units with 0 bw')
    print('model proportion included: ')
    print(np.sum(mt_ix[:])/len(mt_ix[0]))
    print('total: %i' %len(mt_ix[0]))
    print('included: %i' %np.sum(mt_ix[:]))
    print('excluded: %i' %(len(mt_ix[0])-np.sum(mt_ix[:])))
    print('sparse proportion included: ')
    print(np.sum(st_ix[:])/len(st_ix[0]))
    print('total: %i' %len(st_ix[0]))
    print('included: %i' %np.sum(st_ix[:]))
    print('excluded: %i' %(len(st_ix[0])-np.sum(st_ix[:])))
    print('real proportion included: ')
    print(np.sum(rt_ix[:])/len(rt_ix[0]))
    print('total: %i' %len(rt_ix[0]))
    print('included: %i' %np.sum(rt_ix[:]))
    print('excluded: %i' %(len(rt_ix[0])-np.sum(rt_ix[:])))


    mf_bw_pos = mf_bw_pos[mf_ix]
    rf_bw_pos = rf_bw_pos[rf_ix]
    sf_bw_pos = sf_bw_pos[sf_ix]
    mt_bw_pos = mt_bw_pos[mt_ix]
    rt_bw_pos = rt_bw_pos[rt_ix]
    st_bw_pos = st_bw_pos[st_ix]

    mf_bw_neg = mf_bw_neg[mf_ix]
    rf_bw_neg = rf_bw_neg[rf_ix]
    sf_bw_neg = sf_bw_neg[sf_ix]
    mt_bw_neg = mt_bw_neg[mt_ix]
    rt_bw_neg = rt_bw_neg[rt_ix]
    st_bw_neg = st_bw_neg[st_ix]


    xs = np.empty((2,3), dtype=object)
    ys = np.empty((2,3), dtype=object)
    xs[0,:] = [rt_bw_pos, mt_bw_pos, st_bw_pos]
    ys[0,:] = [rt_bw_neg, mt_bw_neg, st_bw_neg]
    xs[1,:] = [rf_bw_pos, mf_bw_pos, sf_bw_pos]
    ys[1,:] = [rf_bw_neg, mf_bw_neg, sf_bw_neg]

    lims = [225,6]
    #Make the scatter plots on the population axes
    for ii in range(xs.shape[0]):
        for jj,ax in enumerate(pop_ax[ii,:]):
            x = xs[ii,jj]
            y = ys[ii,jj]
            clor = clors[jj] # '#444444'
            if use_blobs: 
                pop_ax[ii,jj],_ = plotting_funcs.blobscatter(x,y, ax=ax, **{'facecolor':'none','edgecolor':clor})
            else:
                ax.scatter(x,y)
            plt_kwargs = {'color':'k', 'linestyle':'--', 'alpha':0.8}
            pop_ax[ii,jj].plot(range(lims[ii]+1), **plt_kwargs)
            pop_ax[ii,jj].set_xlim([0,lims[ii]])
            pop_ax[ii,jj].set_ylim([0,lims[ii]])
            if ii==0:
                pop_ax[ii,jj].set_yticks([0, 50, 100, 150, 200])
                pop_ax[ii,jj].set_xticks([0, 50, 100, 150, 200])
                print('x or y>100', sum(np.logical_or(x>100, y>100)), 'out of ', len(x))
            if ii==1:
                pop_ax[ii,jj].set_yticks([0,2,4,6])
                print('x or y>4', sum(np.logical_or(x>4, y>4)), 'out of ', len(x))
            if jj>0:
    #             pop_ax[ii,jj].spines['left'].set_visible(False)
    #             pop_ax[ii,jj].set_yticks([])
                pop_ax[ii,jj].set_yticklabels([])

    inset_lims = [50,1.5]
    inset_ticks = [[25,50], [0.75,1.5]]
    inset_binss = [np.arange(0.1,50,5), np.arange(0.05,2,0.15)]
    #Make insets to show details on first two scatter plots
    for ii in range(xs.shape[0]):
        for jj,ax in enumerate(pop_ax[ii,:2]):
            x = xs[ii,jj]
            y = ys[ii,jj]
            clor = clors[jj] # '#444444' 
            inset_axes_kwargs = {'xlim':[0,inset_lims[ii]], 'ylim':[0,inset_lims[ii]], 
                                 'xticks':inset_ticks[ii], 'yticks':inset_ticks[ii]}
            inset_ax = inset_axes(pop_ax[ii,jj],
                                    width="40%", # width = 30% of parent_bbox
                                    height="40%", # height : 1 inch
                                    loc=1, axes_kwargs=inset_axes_kwargs)

            if use_blobs:
                inset_ax,_ = plotting_funcs.blobscatter(x,y, ax=inset_ax, **{'facecolor':'none','edgecolor':clor})
            else:
                inset_ax.scatter(x,y)
#             inset_ax.hist2d(x,y, bins=inset_binss[ii])
            
#             plt_kwargs = {'color':'k', 'linestyle':'--', 'alpha':0.8}
#             inset_ax.plot(range(lims[ii]+1), **plt_kwargs)

    binss = [np.arange(-2.5,200,5), np.arange(0,6,0.15)]
    #Make the scatter plots on the population axes
    for ii in range(xs.shape[0]):
        bins = binss[ii]
        bincenters = 0.5*(bins[1:]+bins[:-1])
        for jj in range(xs.shape[1]):
            clor = clors[jj]
            plt_kwargs = {'color':clor, 'linestyle':'-', 'alpha':0.8}
            x = xs[ii,jj]
            y = ys[ii,jj]
            xx,binEdges = np.histogram(x, bins = bins)
            yy,binEdges = np.histogram(y, bins = bins)
            hist_ax[ii,0].plot(bincenters, xx/sum(xx), **plt_kwargs)
            hist_ax[ii,1].plot(bincenters, yy/sum(yy), **plt_kwargs)

    hist_ax[0,0].set_yticks([0,0.2,0.4,0.6, 0.8])       
    hist_ax[0,1].set_yticks([0,0.2,0.4,0.6, 0.8])       

    hist_ax[1,0].set_yticks([0,0.1, 0.2, 0.3, 0.4])
    hist_ax[1,1].set_yticks([0,0.1, 0.2, 0.3, 0.4])        

    all_axes = fig.get_axes()
    for ii, ax in enumerate(all_axes):
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')


    pop_ax[0,0].set_ylabel('Temporal span of inhibition (msec)')
    pop_ax[0,0].set_xlabel('Temporal span of excitation (msec)')
    pop_ax[0,1].set_xlabel('Temporal span of excitation (msec)')
    pop_ax[0,2].set_xlabel('Temporal span of excitation (msec)')

    hist_ax[0,0].set_xlabel('Temporal span of excitation (msec)')
    hist_ax[0,1].set_xlabel('Temporal span of inhibition (msec)')
    hist_ax[0,0].set_ylabel('Proportion of units')


    pop_ax[1,0].set_ylabel('Frequency span of inhibition (octave)')
    pop_ax[1,0].set_xlabel('Frequency span of excitation (octave)')
    pop_ax[1,1].set_xlabel('Frequency span of excitation (octave)')
    pop_ax[1,2].set_xlabel('Frequency span of excitation (octave)')

    hist_ax[1,0].set_xlabel('Frequency span of excitation (octave)')
    hist_ax[1,1].set_xlabel('Frequency span of inhibition (octave)')
    hist_ax[1,0].set_ylabel('Proportion of units')

    hist_ax[0,0].set_xticks([0, 50, 100, 150, 200])
    hist_ax[0,1].set_xticks([0, 50, 100, 150, 200])

    if save_path is not None:
        fig.savefig(os.path.join(save_path+'.svg'))