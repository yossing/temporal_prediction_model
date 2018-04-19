import os
import pickle as pkl
import numpy as np
import scipy.misc
import scipy.signal
import scipy.ndimage
from PIL import Image
FLOAT_X = 'float32'

def preprocess_and_finalise_visual_data(vidpath, vid_save_path, final_save_path,
                                 seq_length=50, tensorize_clips=False, max_examples_per_vid=1000, 
                                 filter_type='whiten', filetype='.png', clip_edges=False,
                                 max_examples=20000, random_order=True, verbose=True):

    """
    Convenience function to first preprocess individual videos, saving one at a time and then 
    compile into finalised dataset
    """
    print('preprocessing individual videos...')
    preprocess_vids(vidpath, vid_save_path, 
                    seq_length=seq_length, tensorize_clips=tensorize_clips, 
                    max_examples_per_vid=max_examples_per_vid, 
                    filter_type=filter_type, filetype=filetype, 
                    clip_edges=clip_edges)
    print('Compiling into final dataset...')
    finalise_dataset(vid_save_path, final_save_path, max_examples=20000, random_order=True) 
    print('Done')
    return


def preprocess_vids(vidpath, save_path, n_pixels=180, 
                    seq_length=50, tensorize_clips=False, 
                    max_examples_per_vid=1000, 
                    filter_type='whiten', filetype='.png', 
                    clip_edges=False):
    """
    Loop through all subfolders in directory, where each subfolder contains a seperate movie clip to 
    be preprocessed. Each frame of the movie clip should be a .png or .jpeg image. 
    """

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    folder_names = [x[0] for x in os.walk(vidpath)]
    folder_names = folder_names[1:]
    print(folder_names)
    for folder_name in folder_names:
        print(folder_name)
        filenames = os.listdir(folder_name)
        filenames.sort()
        imlist = []

        for filename in filenames:
            if filename.lower().endswith(filetype) or filename.lower().endswith(filetype) or filename.upper().endswith(filetype):

                im = Image.open(os.path.join(folder_name, filename))
                im = im.convert('L', (0.2989, 0.5870, 0.1140, 0)) #Convert to grayscale
                im = np.array(im)
                im = clip_longest_dim(im)
                im = scipy.misc.imresize(im,(n_pixels,n_pixels))

                if filter_type is not None:
                    if filter_type=='whiten':
                        imw = whiten_and_filter_image(im)
                    elif filter_type=='lowpass': 
                        sigma= 2
                        imw = scipy.ndimage.filters.gaussian_filter(im, sigma)
                    else:
                        imw = im
                else:
                    imw = im
                    
                if clip_edges:
                    start_x = 45
                    end_x = -35
                    start_y = 10
                    end_y = -10
                    imw = imw[start_x:end_x,start_y:end_y]
                imlist.append(imw.astype(FLOAT_X))
        if imlist:
            [d1,d2] = imlist[0].shape
            n_images = len(imlist)
            print(n_images)
            imarr = np.dstack(imlist) #This will give an array of size [d1xd2xn_images]
            print(imarr.shape)
            if tensorize_clips:
                imarr = np.reshape(imarr, [d1*d2, n_images], order='f') #collapse d1 and d2, starting with first axis first
                print(imarr.shape)
                tensarr = tensorize(imarr,seq_length)
                n_stacks = tensarr.shape[-1]
                tensarr = np.reshape(tensarr, [d1,d2,seq_length,n_stacks], order = 'f') #want the first ix to be changed first
            else:
                # n_stacks = int(np.floor(imarr.shape[-1]/seq_length))
                n_stacks = int(np.floor(n_images/seq_length))
                print(n_images)
                print(n_stacks)
                print(n_stacks*seq_length)
                imarr = imarr[:,:,:int(n_stacks*seq_length)]
                #tensarr = np.reshape(imarr, [d1, d2, seq_length, n_stacks])
                tensarr = np.reshape(imarr, [d1, d2, n_stacks, seq_length])
                tensarr = np.rollaxis(tensarr, -1, -2)

            tensarr = np.rollaxis(tensarr,-1) #bring the n_stacks examples to the front
            print(tensarr.shape)

            #Sometimes you migh thave some disproportionally long videos and you only want to 
            #save a limited number of frames from each one to prevent a single video from 
            #dominating the training set. 
            if tensarr.shape[0]>max_examples_per_vid:
                tensarr = tensarr[:max_examples_per_vid,:,:,:]
            #Save preprocessed array 
            pickle_data(tensarr, os.path.join(save_path+os.path.split(folder_name)[-1])+'.pkl')
    
def finalise_dataset(file_path, full_save_path, max_examples = 20000, random_order=True):
    """
    Compile the individually preprocessed movie clips saved in the preprocess_vids function
    into a single array with a given 
    Arguments:
        file_path {string} -- path to folder where individually preprocessed movie clips are saved
        full_save_path {[type]} -- path to location where the final dataset will be saved, ending in .pkl
    
    Keyword Arguments:
        max_examples {int} -- The maximum number of traiing examples to include in the compiled dataset. 
                              If there are fewer exmaples than these, the all of the examples form the preprocessed
                              clips will be included. Otherwise, up to the (default: {'normalized_concattrain.pkl'})
        save_name {str} -- (default: {'normalized_concattrain.pkl'})
        random_order {bool} -- shuffle the exampl order before saving (default: {True})
    """
    pickled_arr_paths = os.listdir(file_path)
    n_pickled_arrs = len(pickled_arr_paths)

    n_arrs_parsed = 0
    n_examples = 0

    example_filename = pickled_arr_paths[0]
    example_arr = load_pickled_data(os.path.join(file_path, example_filename))

    concattrain = np.zeros([max_examples, *example_arr.shape[1:]])
    print('here')
    while n_examples < max_examples and n_arrs_parsed < n_pickled_arrs:
        
        this_filename = pickled_arr_paths[n_arrs_parsed]
        this_arr = load_pickled_data(os.path.join(file_path, this_filename))
        #randomly select example sequences from each movie
        n_entries = this_arr.shape[0]
        select_ix = np.random.permutation(this_arr.shape[0])
        concattrain[n_examples:n_examples+n_entries,:,:,:] = this_arr[select_ix[:n_entries],:,:,:]
        n_examples += n_entries
        n_arrs_parsed +=1

    if random_order:
        perm_seq = np.random.permutation(np.arange(n_examples))
    else:
        perm_seq = np.arange(n_examples)
    concattrain = concattrain[perm_seq,...]

    #normalize by subtracting the mean and dividing by the standard deviation of the whole dataset
    normalized_concattrain = (concattrain - np.mean(concattrain[:]))/np.std(concattrain[:])
    #save the dataset
    pickle_data(normalized_concattrain, full_save_path)
    return 

def load_pickled_data(load_path):
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

def clip_longest_dim(frame):
    [h, w] = frame.shape
    # print('h: %i' %h)
    # print('w: %i' %w)
    shortest_dim = np.minimum(h, w)
    longest_dim = np.maximum(h, w)
    # print(shortest_dim)
    # print(longest_dim)
    start_clip = int(np.round(longest_dim/2) - np.round(shortest_dim/2))
    end_clip = int(start_clip + shortest_dim)
    # print(start_clip)
    # print(end_clip)
    if longest_dim == h:
        clip_frame = frame[start_clip:end_clip, :]
    else:
        clip_frame = frame[:, start_clip:end_clip]
    # print(clip_frame.shape)
    return clip_frame

def whiten_and_filter_image(im_to_filt):
    N = im_to_filt.shape[0]
    imf=np.fft.fftshift(np.fft.fft2(im_to_filt))
    f=np.arange(-N/2,N/2)
    [fx, fy] = np.meshgrid(f,f)
    [rho,theta]=cart2pol(fx,fy)
    filtf = rho*np.exp(-0.5*(rho/(0.7*N/2))**2)
    imwf = filtf*imf
    imw = np.real(np.fft.ifft2(np.fft.fftshift(imwf)))
    return imw

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def tensorize(X_arr, n_h, lag=0, remove_zeros = True, float_X = 'float32'):
    #Modified from Benware  
    # Add a history dimension to a 2D stimulus grid
    # Inputs:
    #  X_arr -- stimulus, freq x time
    #  n_h -- number of history steps
    #  lag -- minimum lag
    # 
    # Outputs:
    #  X_tens -- stimulus, freq x history x time

    n_d1 = np.shape(X_arr)[0]
    n_d2 = np.shape(X_arr)[1]

    # pad with zeros
    # X_arr = np.concatenate(np.zeros((n_d1, n_h)), X_arr)

    n_d2_pad = np.shape(X_arr)[1]

    # preallocate
    X_tens = np.zeros((n_d1, n_h, n_d2_pad), dtype=float_X);

    for ii in range(n_h):
        X_tens[:,ii,:] = shift(X_arr, (0,lag+n_h-ii-1))#.reshape(n_d1, 1, n_d2_pad)

    if remove_zeros:
        # X_tens = X_tens[:, :, n_h+1:]
        X_tens = X_tens[:, :, n_h-1:]
    return X_tens
