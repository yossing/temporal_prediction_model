'''
Functions for visualisation of networks and costs
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import quantify_aud_strfs as quant
from imp import reload
import matplotlib.gridspec as gridspec
reload(quant)

FLOATX='float32' # needed to use the GPU

#########################################################################################
##############################           General          ###############################
#########################################################################################

def plot_loss(train_losses, val_losses=None, title=None):
    '''
       Plot cost history of training and optionally of validation
       on a loglog axis
    '''
    matplotlib.rcParams['figure.figsize'] = (8.0, 6.0)
    num_train_epochs = len(train_losses)

    train_epochs = np.arange(num_train_epochs)
    plt.loglog(train_epochs, train_losses, label='train loss')
    if val_losses is not None and val_losses != []:
        val_epochs = np.arange(len(val_losses))
        plt.loglog(val_epochs, val_losses, label='val loss')
    plt.legend()
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    if title is None:
        plt.title('Loss over Training Epochs')
    else:
        plt.title(title)
    plt.show()


#########################################################################################
##############################             Visual          ##############################
#########################################################################################

    
def getVisualWeightImageFromNetworkParams(network_params, RF_size, clip_length, sort=False):
    weights = network_params[0].T
    keep_ix = np.sum(weights**2,1)>0.01*np.max(np.sum(weights**2,1))
    weights = weights[keep_ix,:]
    if sort:
        l2_norm = np.sum(weights**2,1)
        sort_ix = np.argsort(l2_norm)
        weights = weights[sort_ix,:]
    numweights = weights.shape[0]
    # print(weights.shape)
    weights = np.reshape(weights, [numweights,clip_length, RF_size,RF_size])
    weights = np.rollaxis(weights,2,1)
    weights = np.rollaxis(weights,3,2)
    #     print(weights.shape)

    dispgrid_size = int(np.ceil(np.sqrt(numweights)))
    weight_image = np.zeros([dispgrid_size*RF_size,dispgrid_size*RF_size,clip_length])

    for ii in np.arange(numweights):
        ii+=1
        x_offset = int(RF_size*np.mod(ii-1,dispgrid_size)+1) -1
        y_offset = int(RF_size*np.floor((ii-1)/dispgrid_size)+1) -1
        this_weight = weights[ii-1,...]
        this_weight = this_weight/np.max(np.abs(this_weight[:]))
        weight_image[y_offset:y_offset+(RF_size-1)+1, x_offset:x_offset+(RF_size-1)+1,:] = this_weight
    return weight_image

def getVisualWeightImage(weights, RF_size, clip_length, sort=False, keep_prop = None, order = 0, normalize_weights=True, verbose=False):

    w_shape = weights.shape
    weights = np.reshape(weights, [w_shape[0], np.prod(w_shape[1:])])

    if keep_prop is not None:

        keep_ix = np.sum(weights**2,1)>keep_prop*np.max(np.sum(weights**2,1))
        # print(keep_ix.shape)
        weights = weights[keep_ix,...]
    if sort:
        l2_norm = np.sum(weights**2,1)
        sort_ix = np.argsort(l2_norm)
        # print(sort_ix.shape)
        weights = weights[sort_ix,...]

    # weights = np.reshape(weights, [-1, *w_shape[1:]])

    numweights = weights.shape[0]
    if verbose:
        print(weights.shape)

    if order == 0:
        weights = np.reshape(weights, [numweights,clip_length, RF_size,RF_size])
        weights = np.rollaxis(weights,2,1)
        weights = np.rollaxis(weights,3,2)
    else:
        weights = np.reshape(weights, [numweights,RF_size,RF_size, clip_length])
    if verbose:
        print(weights.shape)

    dispgrid_size = int(np.ceil(np.sqrt(numweights)))
    weight_image = np.zeros([dispgrid_size*RF_size,dispgrid_size*RF_size,clip_length])

    for ii in np.arange(numweights):
        x_offset = int(RF_size*np.mod(ii,dispgrid_size)+1) -1
        y_offset = int(RF_size*np.floor((ii)/dispgrid_size)+1) -1
        this_weight = weights[ii,:,:,:]
        if normalize_weights:
            this_weight = this_weight/np.max(np.abs(this_weight[:]))
        weight_image[y_offset:y_offset+(RF_size-1)+1, x_offset:x_offset+(RF_size-1)+1,:] = this_weight
    return weight_image

def plot_temporal_seq(in_seq, reshape_params=None, t_keep=None, order='c', outer_gridspec=None, figsize=[8,8], fig=None, normalize_weights=True):
    if reshape_params:
        seq = np.reshape(in_seq, [in_seq.shape[0], *reshape_params], order=order)
    else:
        seq = in_seq
    if t_keep is not None:
        # t_keep = seq.shape[-1]
        seq = seq[...,t_keep]
        
    grid_w = seq.shape[-1]
    grid_h = seq.shape[0]

    if outer_gridspec is not None:
        inner_grid = gridspec.GridSpecFromSubplotSpec(grid_h, grid_w,
            subplot_spec=outer_gridspec, wspace=0.01, hspace=0.00)
    else:
        fig = plt.figure(figsize=figsize)
        inner_grid = gridspec.GridSpec(grid_h, grid_w, wspace=0.01, hspace=0.00)
    cnt = 0 
    # vmax = np.max(np.abs(seq.flatten()))
    for ii in range(grid_h):
        vmax = np.max(np.abs(seq[ii,...].flatten()))
        for jj in range(grid_w):
            ax = plt.Subplot(fig, inner_grid[cnt])
            # ax.imshow(seq[ii,:,:,jj], cmap='gray', origin='lower')
            if normalize_weights:
                ax.imshow(seq[ii,:,:,jj], cmap='gray', vmin=-vmax, vmax=vmax, origin='lower')#, origin='upper')
            else:
                ax.imshow(seq[ii,:,:,jj], cmap='gray')#, vmin=-vmax, vmax=vmax, origin='lower')#, origin='upper')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
            cnt+=1

    return ax

def plot_vis_weights_tsteps(network_params, RF_size=20, clip_length=7, keep_prop = 0.01, n_tsteps=7, n_weights = 10, figsize=(12,12)):
    (W_i, b_i, W_o, b_o)  = network_params

    l2_norm = np.sum(W_i**2,axis=0);
    keep_ix = l2_norm>keep_prop*max(l2_norm)
    W_i = W_i[...,keep_ix] 
    W_o = W_o[keep_ix,...]

    l2_norm = l2_norm[keep_ix]
    sort_ix = np.argsort(l2_norm);

    W_i = W_i[...,sort_ix] 
    W_o = W_o[sort_ix,...]
    weights = W_i
    numweights = weights.shape[1]
    weights = np.reshape(weights.T, [numweights,clip_length, RF_size,RF_size])
    weights = np.rollaxis(weights,2,1)
    weights = np.rollaxis(weights,3,2)

    plt.figure(figsize=figsize)
    rand_ix = np.random.permutation(numweights)
    for ii in range(n_weights):
        for jj in range(n_tsteps):
            plt.subplot(n_tsteps,n_weights,(jj*n_weights) + ii+1)
            ax = plt.gca()
            ax.imshow(np.reshape(weights[rand_ix[ii],:,:,jj],[RF_size,RF_size]), cmap='gray', origin='lower')
            ax.axis('off')
    plt.show()

def plot_these_vis_weights(weights, RF_size=20, clip_length=7, keep_prop = 0.01, tstep = None, figsize=(5,5), order = 0, ax=None, normalize_weights=True, sort=False):
    # weight_image = getVisualWeightImageFromNetworkParams(weights, RF_size, clip_length, order=order)
    weight_image = getVisualWeightImage(weights, RF_size, clip_length, order=order, normalize_weights=normalize_weights, sort=sort, keep_prop=keep_prop)
    if tstep is None:
        for tstep in range(clip_length):
            
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
            ax.imshow(weight_image[:,:,tstep], cmap='gray', origin='lower')
            ax.set_xticks(np.arange(-0.5,weight_image.shape[0]-0.5, RF_size))
            ax.set_yticks(np.arange(-0.5,weight_image.shape[1]-0.5, RF_size))
            plt.grid()
            plt.show()
    else:
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
        ax.set_xticks(np.arange(-0.5,weight_image.shape[0]-0.5, RF_size))
        ax.set_yticks(np.arange(-0.5,weight_image.shape[1]-0.5, RF_size))
        ax.imshow(weight_image[:,:,tstep], cmap='gray', origin='lower')
        ax.grid()
        # plt.show()

#########################################################################################
##############################           Auditory          ##############################
#########################################################################################

def getWeightImage(weights,num_freq,n_h,Who=None, half_t=False, keep_prop=None, flip_signs=True):


    # weights = weights.transpose()

    if len(weights.shape)==2:
        if keep_prop is not None:
            l2_norm = np.sum(weights**2,axis=1)#*np.sum(W_o**2,axis=1)
            keep_ix = l2_norm>keep_prop*max(l2_norm)
            weights = weights[keep_ix,:] 

            l2_norm = l2_norm[keep_ix]
            sort_ix = np.argsort(l2_norm);

            weights = weights[sort_ix,:] 

        weights = weights.reshape(-1, n_h, num_freq)
        weights = np.rollaxis(weights, 2, 1)

    numweights = weights.shape[0]
    # # weights = weights[:,:,int(np.ceil(n_h/2)):]
    # # n_h = int(np.ceil(n_h/2))
    if half_t:
        print("This is the end")

    else:
        dispgrid_size = int(np.ceil(np.sqrt(numweights)))
        weight_image = np.zeros((dispgrid_size*num_freq,dispgrid_size*n_h))
        weight_image_norm = np.zeros((dispgrid_size*num_freq,dispgrid_size*n_h))

    if flip_signs:
        weights = quant.flip_neg_weights(weights)

    for ii in range(numweights):
                   
        x_offset = int(n_h*np.mod(ii,dispgrid_size))
        y_offset = int(num_freq*np.floor((ii)/dispgrid_size))
        this_weight = weights[ii,:,:] 

        this_weight_norm = this_weight/max(abs(this_weight.flatten()))
        # this_weight_norm = this_weight/np.sum(this_weight[:]**2)


        # sign_tlast = np.sign(sum(this_weight[:,-1]))
        # sign_output = np.sign(sum(Who[1:np.shape(this_weight)[1],ii]))
        # if (sign_output != sign_tlast):
        #     this_weight = -this_weight
#             if max(this_weight(:))<max(abs(this_weight(:)))
#                 this_weight = -this_weight;

        weight_image[y_offset:y_offset+num_freq, x_offset:x_offset+n_h] = this_weight
        weight_image_norm[y_offset:y_offset+num_freq, x_offset:x_offset+n_h] = this_weight_norm
    # print((weight_image == weight_image_norm).sum())

    return weight_image, weight_image_norm

def plot_these_aud_weights(network_params, n_h=40, num_freq=32, keep_prop = 0.01, figsize=(12,12), ax=None, half_t = False, flip_signs=True):
# n_h = 32
# num_freq = 40
    (W_i, b_i, W_o, b_o)  = network_params

    l2_norm = np.sum(W_i**2,axis=0)#*np.sum(W_o**2,axis=1)
    keep_ix = l2_norm>keep_prop*max(l2_norm)
    W_i = W_i[:,keep_ix] 
    # W_o = W_o[keep_ix,:]

    l2_norm = l2_norm[keep_ix]
    sort_ix = np.argsort(l2_norm);

    W_i = W_i[:,sort_ix] 
    # W_o = W_o[sort_ix,:]

    weights = W_i
    weights = weights.transpose()

    numweights = weights.shape[0]

    weights = weights.reshape(numweights, n_h, num_freq)
    weights = np.rollaxis(weights, 2, 1)

    if half_t:
        weights = weights[:,:,int(np.ceil(n_h/2)):]
        n_h = int(np.ceil(n_h/2))


    [weight_image, weight_image_norm] = getWeightImage(weights,num_freq,n_h, half_t, flip_signs=flip_signs)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    ax.set_xticks(np.arange(0,weight_image_norm.shape[1], n_h))
    ax.set_yticks(np.arange(0,weight_image_norm.shape[0], num_freq))
    
    # h_i_max = np.amax(abs(weight_image.flatten()))
    # im = plt.imshow(weight_image, cmap='gray', vmin=-h_i_max, vmax = h_i_max)
    h_i_max = np.amax(abs(weight_image_norm.flatten()))
    im = plt.imshow(weight_image_norm, cmap='seismic', vmin=-h_i_max, vmax = h_i_max)
    plt.grid()
    #plt.show()
    return [ax, im]

