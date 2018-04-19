"""
Functions to load training data. 
"""

import numpy as np
import ntpath
import tables
import sys
import os
import pickle as pkl

FLOATX = 'float32'
RANDOM_SEED = 12345


#***************************************************************************************************
#****************************************GENERAL FUNCTIONS******************************************
#***************************************************************************************************

def load_pickled_data(load_path):
    load_path = os.path.expanduser(load_path)
    with open(load_path, "rb") as f:
        dat = pkl.load(f)
    return dat

def pickle_data(data, save_path, protocol=4):
    #Convenience funciton to save data as pickle. 
    #Create the file directory (including parents) is it doesn't exist.
    if not os.path.exists(os.path.dirname(os.path.abspath(save_path))):
        os.makedirs(os.path.dirname(os.path.abspath(save_path)))
    with open(save_path, "wb") as f:
        pkl.dump(data, f, protocol=protocol)
    return

def add_noise(data, noise_ratio, renormalise=True, set_seed=True):
    if set_seed:
        np.random.seed(RANDOM_SEED)
    std_data = data.std()
    out_data = data + std_data*noise_ratio*np.random.randn(*data.shape)

    if renormalise:
        new_mean_data = np.mean(out_data[:])
        new_std_data = np.std(out_data[:])
        out_data = (out_data - new_mean_data)/new_std_data

    return out_data

def extract_train_val_data(concattrain, val_prop=0.1, random_order=False, set_seed=False):
    tot_train_size = concattrain.shape[0]

    val_size = int(tot_train_size*val_prop)
    train_size = tot_train_size - val_size
    if random_order:
        if set_seed:
            np.random.seed(RANDOM_SEED)
        perm_seq = np.random.permutation(np.arange(tot_train_size))
    else:
        perm_seq = np.arange(tot_train_size)

    train_data = concattrain[perm_seq[:train_size], ...]
    val_data = concattrain[perm_seq[-val_size:], ...]

    return train_data, val_data


def on_the_fly_preprocessing(X_train, y_train, 
                             X_val, y_val, 
                             noise_ratio=0, 
                             input_noise_ratio=0,  
                             max_examples=20000, 
                             copy_data=True, 
                             norm_type=0):
    """
    Do all of the data manipulations that are not performed in the original preprocessing. 
    It is useful to have some of the preprocessing operations can be treated as hyperparameters. 

    Arguments:
        X_train {numpy array} 
        y_train {numpy array} 
        X_val {numpy array} 
        y_val {numpy array} 
    
    Keyword Arguments:
        noise_ratio {number} -- noise to signal ratio of noise to add to both inputs and outputs (default: {0})
        input_noise_ratio {number} -- oise to signal ratio of noise to add to both inputs (default: {0})
        max_examples {number} -- max number of training examples. Will discard remaining (default: {20000})
        copy_data {bool} -- whether to return a c-ordered copy of the data after preprocessing (default: {True})
        norm_type {number} -- normalise the whole dataset at once or each example seperately (default: {0})
    """

    if X_train.shape[0] > max_examples:
        X_train = X_train[:max_examples, ...]
        y_train = y_train[:max_examples, ...]
        X_val = X_val[:max_examples//10, ...]
        y_val = y_val[:max_examples//10, ...]

    if input_noise_ratio != 0 and input_noise_ratio is not None:
        assert(noise_ratio==0)
        print('NB! Adding input only noise with noise_ratio = %.2f' %input_noise_ratio)
        X_train = add_noise(X_train, input_noise_ratio, renormalise=False, set_seed=True)
        # X_val = add_noise(X_val, input_noise_ratio, renormalise=False)

    elif noise_ratio != 0 and noise_ratio is not None:
        assert(input_noise_ratio==0)
        print('NB! Adding noise with noise_ratio = %.2f' %noise_ratio)
        X_train = add_noise(X_train, noise_ratio, renormalise=False, set_seed=True)
        # X_val = add_noise(X_val, noise_ratio, renormalise=False)
        y_train = add_noise(y_train, noise_ratio, renormalise=False, set_seed=False)
        # y_val = add_noise(y_val, noise_ratio, renormalise=False)

    if norm_type == 0:
    #Normalise by subtracting the mean and dividing by the standard deviation of the entire dataset
        X_train_mean = X_train.mean()
        X_train_std = X_train.std()
        X_train = ((X_train-X_train_mean)/X_train_std)
        y_train = ((y_train-X_train_mean)/X_train_std)
        X_val = ((X_val-X_train_mean)/X_train_std)
        y_val = ((y_val-X_train_mean)/X_train_std)

    else:
    #Normalise by subtracting the mean and dividing by the standard deviation of each example seperately
        X_train_mean = np.reshape(X_train, [X_train.shape[0], -1]).mean(axis=-1)
        X_train_std = np.reshape(X_train, [X_train.shape[0], -1]).std(axis=-1)

        X_train = (X_train-X_train_mean[:, np.newaxis, np.newaxis])/X_train_std[:, np.newaxis, np.newaxis]
        y_train = (y_train-X_train_mean[:, np.newaxis, np.newaxis])/X_train_std[:, np.newaxis, np.newaxis]
        
        X_val_mean = np.reshape(X_val, [X_val.shape[0], -1]).mean(axis=-1)
        X_val_std = np.reshape(X_val, [X_val.shape[0], -1]).std(axis=-1)
        
        X_val = (X_val-X_val_mean[:, np.newaxis, np.newaxis])/X_val_std[:, np.newaxis, np.newaxis]
        y_val = (y_val-X_val_mean[:, np.newaxis, np.newaxis])/X_val_std[:, np.newaxis, np.newaxis]

    if copy_data:
        #This is memory intensive, but helps to ensure that the data is stored in c-ordered arrays
        #even after all of the reshaping and shuffling. This runs much faster during training. 
        X_to_train = X_train.copy(order='c').astype('float32')
        y_to_train = y_train.copy(order='c').astype('float32')
        X_to_val = X_val.copy(order='c').astype('float32')
        y_to_val = y_val.copy(order='c').astype('float32')
    else:
        X_to_train = X_train.astype('float32')
        y_to_train = y_train.astype('float32')
        X_to_val = X_val.astype('float32')
        y_to_val = y_val.astype('float32')

    print(X_to_train.shape)
    print(y_to_train.shape)
    print(X_to_val.shape)
    print(y_to_val.shape)
    
    return [X_to_train, y_to_train, X_to_val, y_to_val]

#***************************************************************************************************
#***************************FUNCTIONS FOR LOADING VISUAL DATASET************************************
#***************************************************************************************************

def divide_rfs(dat_to_divide, RF_size):
    """
    Take the maximimum number of RF_size examples from each full frame. 
    Reshape the visual dataset from shape: [n_examples, x1, x2, t] to [n_examples_tot, RF_size*RF_size, t]
    
    Arguments:
        dat_to_divide {numpy array} -- the preprocessed visual data to be reshaped
        RF_size {int} -- number of pixels in each spatial dimension
    
    Returns:
        [numpy array] -- [reshaped data]
    """
    def divide_along_dim_1(temp):
        [x1, x2, t, m] = temp.shape
        nrf = int(np.floor(x1/RF_size))
        temp = np.reshape(temp, [RF_size, nrf, x2, t, m], order='f')
        temp = np.moveaxis(temp, 1, -1)
        temp = np.reshape(temp, [RF_size, x2, t, m*nrf], order='c')
        temp = np.rollaxis(temp, 1, 0)
        return temp
    #start with data in the format: [m, x1, x2, t] eg [914,160,100,50]
    #move examples to last dimension
    dat_to_divide = np.moveaxis(dat_to_divide, 0, -1)
    #dat_to_divide.shape: [x1,x2,t,m]
    #divide second spatial dimension into RF_size chunks
    dat_to_divide = divide_along_dim_1(dat_to_divide)
    #dat_to_divide.shape: [x1,RF_size,t,mm]
    #Do the same for the first spatial dimension
    dat_to_divide = divide_along_dim_1(dat_to_divide)
    #dat_to_divide.shape: [RF_size,RF_size,t,mmm]
    #collapse spatial dimensions into single vector
    [_, _, seq_length, m] = dat_to_divide.shape
    dat_to_divide = np.reshape(dat_to_divide, [RF_size*RF_size, seq_length, m], order='f')
    #dat_to_divide.shape: [RF_size*RF_size,t,mmm]
    #move examples to first dimension
    dat_to_divide = np.rollaxis(dat_to_divide, -1)
    #train_data.shape: [mmm,RF_size*RF_size,t]
    return dat_to_divide

def load_1d_conv_vis_data(data_path, 
                          noise_ratio=0, 
                          input_noise_ratio=0,  
                          max_examples=20000, 
                          copy_data=True, 
                          norm_type=0, 
                          RF_size=20, 
                          t_filter_length=7, 
                          t_predict_length=1):
    """"
    
    Load preprocessed visual dataset and format it for inputing into 1D convolutional network. 

    Arguments:
        data_path {string} -- Full path to the preprocessed dataset. This must be saved as a pickle
    
    Keyword Arguments:
        noise_ratio {number} -- noise to signal ratio of noise to add to both inputs and outputs (default: {0})
        input_noise_ratio {number} -- oise to signal ratio of noise to add to both inputs (default: {0})
        max_examples {number} -- max number of training examples. Will discard remaining (default: {20000})
        copy_data {bool} -- whether to return a c-ordered copy of the data after preprocessing (default: {True})
        norm_type {number} -- normalise the whole dataset at once or each example seperately (default: {0})
        RF_size {int} -- Number of pixels in each spatial dimension (default: {20})
        t_filter_length {number} -- The number of input timesteps === length of temporal filter (default: {7})
        t_predict_length {number} -- Number of time steps to be predicted (default: {1})
    """

    filedir, filename = ntpath.split(data_path)
    if filename == '':
        filename = 'normalized_concattrain.pkl'

    concattrain = load_pickled_data(os.path.join(filedir, filename))
    concattrain = divide_rfs(concattrain, RF_size)
    [train_data, val_data] = extract_train_val_data(concattrain, random_order=False)
    del concattrain

    #Select appropriate timesteps for 1D temporal convolution
    print('t_filter_length: ', t_filter_length)
    print('t_predict_length: ', t_predict_length)
    X_train = train_data[:, :, :-t_predict_length]
    y_train = train_data[:, :, t_filter_length:]
    X_val = val_data[:, :, :-t_predict_length]
    y_val = val_data[:, :, t_filter_length:]
    del train_data
    del val_data
    [X_to_train, y_to_train, X_to_val, y_to_val] = on_the_fly_preprocessing(X_train, y_train, 
                                                                            X_val, y_val, 
                                                                            noise_ratio=noise_ratio, 
                                                                            input_noise_ratio=input_noise_ratio,  
                                                                            max_examples=max_examples, 
                                                                            copy_data=copy_data, 
                                                                            norm_type=norm_type)
    return [X_to_train, y_to_train, X_to_val, y_to_val]


def load_tensorized_visual_data_for_fcn(data_path, 
                                        noise_ratio=0, 
                                        input_noise_ratio=0, 
                                        t_past=7, t_future=1, 
                                        copy_data=False, 
                                        RF_size=None, 
                                        norm_type=0, 
                                        max_examples=500000):

    """
    Load preprocessed visual dataset and format it for inputing into fully connected network. 
    Ideally this should be done on tensorized data. This can eaily be done by setting the 
    'tensorized' keyword to True in preprocess_vids

    """


    filedir, filename = ntpath.split(data_path)
    if filename == '':
        filename = 'normalized_concattrain.pkl'

    concattrain = load_pickled_data(os.path.join(filedir, filename))

    if RF_size is not None:
        concattrain = divide_rfs(concattrain, RF_size)
    [train_data, val_data] = extract_train_val_data(concattrain, random_order=False)

    X_train = train_data[..., :t_past]
    y_train = np.squeeze(train_data[..., t_past:t_past+t_future])
    X_val = val_data[..., :t_past]
    y_val = np.squeeze(val_data[..., t_past:t_past+t_future])

    X_train = np.rollaxis(X_train, -1, -2)
    X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1]*X_train.shape[2]])
    X_val = np.rollaxis(X_val, -1, -2)
    X_val = np.reshape(X_val, [X_val.shape[0], X_val.shape[1]*X_val.shape[2]])
    
    if t_future > 1:
        y_train = np.rollaxis(y_train, -1, -2)
        y_train = np.reshape(y_train, [y_train.shape[0], y_train.shape[1]*y_train.shape[2]])
        y_val = np.rollaxis(y_val, -1, -2)
        y_val = np.reshape(y_val, [y_val.shape[0], y_val.shape[1]*y_val.shape[2]])


    [X_to_train, y_to_train, X_to_val, y_to_val] = on_the_fly_preprocessing(X_train, y_train,
                                                                            X_val, y_val, 
                                                                            noise_ratio=noise_ratio, 
                                                                            input_noise_ratio=input_noise_ratio,  
                                                                            max_examples=max_examples, 
                                                                            copy_data=copy_data, 
                                                                            norm_type=norm_type)
    return [X_to_train, y_to_train, X_to_val, y_to_val]


#***************************************************************************************************
#********************************FUNCTIONS FOR LOADING AUDITORY DATASET*****************************
#***************************************************************************************************

def load_matfile_dataset(mat_path):
    with tables.open_file(mat_path) as f:
        dataset = np.asarray(f.root.concattrain[:], dtype=FLOATX)
    return dataset

def reshape_auditory_input_data(data, t_past, t_future, numfreq):
    I = t_past*numfreq
    K = t_future*numfreq 
    x = data[:, :I]
    y = data[:, I:I+K]
    return x, y

def load_auditory_data(file_path, t_past, t_future, numfreq, 
                       post_dict=False, 
                       noise_ratio=0, 
                       input_noise_ratio=0):
    """
    
    
    Arguments:
        file_path {string} -- [description]
        t_past {[type]} -- [number of past time steps to enter into network]
        t_future {[type]} -- [number of future timesteps to predict]
        numfreq {[type]} -- [number of frequencies ]
    
    Keyword Arguments:
        post_dict {bool} -- [description] (default: {False})
        noise_ratio {number} -- [description] (default: {0})
        input_noise_ratio {number} -- [description] (default: {0})
    
    Returns:
        [type] -- [description]
    """

    concattrain = load_matfile_dataset(file_path)
    
    if post_dict is True:
        print('NB! post-diction ordering rather than prediction!!!')
        concattrain = concattrain[:, ::-1] #reverse order of 

    train_data, val_data = extract_train_val_data(concattrain, random_order=True, set_seed=True)
    X_train, y_train = reshape_auditory_input_data(train_data, t_past, t_future, numfreq)
    X_val, y_val = reshape_auditory_input_data(val_data, t_past, t_future, numfreq)
    
    on_the_fly_preprocessing(X_train, y_train, 
                             X_val, y_val, 
                             noise_ratio=noise_ratio, 
                             input_noise_ratio=input_noise_ratio,  
                             max_examples=500000, 
                             copy_data=True, 
                             norm_type=0)
    
    return X_train, y_train, X_val, y_val
