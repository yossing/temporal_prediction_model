import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size    

def plot_2D_projection(Y, ax, n_examples = 100, labels = None, clors = None, markers = None):
    n_plots = int(Y.shape[0]/n_examples)
    for ii in range(n_plots):
        if labels is not None:
            label = labels[ii]
        else:
            label = ''   
        if clors is not None:
            clor = clors[ii]
        else:
            clor = np.random.rand(3,1)
        if markers is not None:
            marker = markers[ii]
        else:
            marker = 'o'    
        alpha = 1
        if marker == 'o' or marker == '^':
            ax.scatter(Y[ii*n_examples:(ii+1)*n_examples, 0], Y[ii*n_examples:(ii+1)*n_examples, 1], edgecolor = clor, facecolors = 'none', marker=marker, alpha=alpha, label=label)
        # elif marker == 'blob':
        #     plt_kwargs = {'edgecolor':clor, 'facecolors':'none', 'marker':'o', 'alpha':alpha, 'label':label}
        #     blobscatter(Y[ii*n_examples:(ii+1)*n_examples, 0], Y[ii*n_examples:(ii+1)*n_examples, 1], ax=ax, **plt_kwargs)
        else:
            ax.scatter(Y[ii*n_examples:(ii+1)*n_examples, 0], Y[ii*n_examples:(ii+1)*n_examples, 1], edgecolor = clor, facecolors = clor, marker=marker, alpha=alpha, label=label)


    if labels is not None:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
    return ax

def plot_2D_projection_with_hists(Y, axScatter, n_examples = 100, labels = None, clors = None, markers = None):

    # create new axes on the right and on the top of the current axes.
    divider = make_axes_locatable(axScatter)
    
    aspect = 4
    pad_fraction = 0.05
    width = axes_size.AxesY(axScatter, aspect=1/aspect)
    pad = axes_size.Fraction(pad_fraction, width)
    axHistx = divider.append_axes("top", size=width, pad=0.1, sharex=axScatter)
    axHisty = divider.append_axes("right", size=width, pad=0.1, sharey=axScatter)
    lim = 20
    binwidth = 1
    bins = np.arange(-lim, lim + binwidth, binwidth)
    linewidth = 3
    

    bincenters = 0.5*(bins[1:]+bins[:-1])
    n_plots = int(Y.shape[0]/n_examples)
    for ii in range(n_plots):
        if labels is not None:
            label = labels[ii]
        else:
            label = ''   
        if clors is not None:
            clor = clors[ii]
        else:
            clor = np.random.rand(3,1)
        if markers is not None:
            marker = markers[ii]
        else:
            marker = 'o'    
        alpha = 1
        if marker == 'o' or marker == '^':
            axScatter.scatter(Y[ii*n_examples:(ii+1)*n_examples, 0], Y[ii*n_examples:(ii+1)*n_examples, 1], edgecolor = clor, facecolors = 'none', marker=marker, alpha=alpha, label=label)
        # elif marker == 'blob':
        #     print('using blob scttaer')
        #     plt_kwargs = {'edgecolor':clor, 'facecolors':'none', 'alpha':alpha, 'label':label}
        #     blobscatter(Y[ii*n_examples:(ii+1)*n_examples, 0], Y[ii*n_examples:(ii+1)*n_examples, 1], ax=axScatter, **plt_kwargs)
        else:
            axScatter.scatter(Y[ii*n_examples:(ii+1)*n_examples, 0], Y[ii*n_examples:(ii+1)*n_examples, 1], edgecolor = clor, facecolors = clor, marker=marker, alpha=alpha, label=label)
                
        xx,binEdges = np.histogram(Y[ii*n_examples:(ii+1)*n_examples, 0], bins = bins, density=1)
        yy,binEdges = np.histogram(Y[ii*n_examples:(ii+1)*n_examples, 1], bins = bins, density=1)
        
        plt_kwargs = {'color':clor, 'linewidth':linewidth, 'alpha':alpha}
        
        axHistx.plot(bincenters,xx, **plt_kwargs)
        axHisty.plot(yy, bincenters, **plt_kwargs)
    
    if labels is not None:
        handles, labels = axScatter.get_legend_handles_labels()
        axScatter.legend(handles, labels)
    return [axScatter, axHistx,axHisty]

def plot_scatter_hist(x, y, axScatter, bins, hist_axes = None,  label = None, clor = None, marker = 'o', alpha=1, n_examples=None):

    # create new axes on the right and on the top of the current axes.
    if hist_axes is None: 
        divider = make_axes_locatable(axScatter)
        aspect = 4
        pad_fraction = 0.05
        width = axes_size.AxesY(axScatter, aspect=1/aspect)
        pad = axes_size.Fraction(pad_fraction, width)
        axHistx = divider.append_axes("top", size=width, pad=0.1, sharex=axScatter)
        axHisty = divider.append_axes("right", size=width, pad=0.1, sharey=axScatter)
    else:
        axHistx = hist_axes[0]
        axHisty = hist_axes[1]

    linewidth = 3  
    bincenters = 0.5*(bins[1:]+bins[:-1])

    if n_examples is not None: 
        ix = np.arange(len(x))
        random.shuffle(ix)
        ix = ix[:n_examples]
    else:
        ix = np.arange(len(x))
    
    binwidth = bins[1]-bins[0]
    x_jitter = 0.5*binwidth*(2*np.random.rand(len(ix)) - 1)
    y_jitter = 0.5*binwidth*(2*np.random.rand(len(ix)) - 1)
#     x_jitter = 0
#     y_jitter = 0
    if marker == 'o':
        axScatter.scatter(x[ix]+x_jitter, y[ix]+y_jitter, edgecolor = clor, facecolors = 'none', marker=marker, alpha=alpha, label=label)
    # elif marker == 'blob':
    #     if clor == 'black' or clor == 'k':
    #         facecolors = clor
    #     else:
    #         facecolors = 'none'    
    #     plt_kwargs = {'edgecolor':clor, 'facecolors':facecolors, 'marker':'o', 'alpha':alpha, 'label':label, 'linewidth':2}
    #     blobscatter(x, y, ax=axScatter, minsize=10*binwidth, **plt_kwargs)
    else:
        axScatter.scatter(x[ix]+x_jitter, y[ix]+y_jitter, edgecolor = clor, facecolors = clor, marker=marker, alpha=alpha, label=label)

    xx,binEdges = np.histogram(x, bins = bins, density=1)
    yy,binEdges = np.histogram(y, bins = bins, density=1)
    plt_kwargs = {'color':clor, 'linewidth':linewidth, 'alpha':alpha}
    axHistx.plot(bincenters,xx, **plt_kwargs)
    axHisty.plot(yy, bincenters, **plt_kwargs)

    return [axScatter, axHistx, axHisty]
