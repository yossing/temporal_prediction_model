from imp import reload
import pickle as pkl
import numpy as np 
import numpy.linalg as linalg
# import scipy.linalg as linalg
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import auditory_analysis as aan
import copy as cp
reload(aan)

def getPeaksAndBWs(strf,dt=5,df=1/6, discard_thresh=0.05):

    original_strf= strf

    strf=np.maximum(original_strf,0)
    l2_norm_pos = np.sum(strf[:]**2)

    [u,s,v] = linalg.svd(strf)  

    f1 = u[:,0]
    t1 = v.T[:,0]

    abs_max_f1_val = np.max(np.abs(f1))
    abs_max_f1_ix = np.argmax(np.abs(f1)) 
    abs_max_t1_val = np.max(np.abs(t1)) 
    abs_max_t1_ix = np.argmax(np.abs(t1)) 

    pos_peaks_ix = np.argwhere(np.abs(t1)>0.1*abs_max_t1_val)
    if len(pos_peaks_ix)>1:
        pos_first_peak_ix = pos_peaks_ix[-1]
    else:
        pos_first_peak_ix = pos_peaks_ix
        
    f_pos_peak = (abs_max_f1_ix)*df
    f_pos_bw = np.sum(np.abs(f1)>0.5*abs_max_f1_val)*df
    t_pos_peak = (len(t1) - abs_max_t1_ix)*dt*-1
    t_pos_bw = np.sum(np.abs(t1)>0.5*abs_max_t1_val)*dt


    #Inhibition:
    strf=np.minimum(original_strf,0)
    l2_norm_neg = np.sum(strf[:]**2)

    [u,s,v] = linalg.svd(strf)  
    f1 = u[:,0]
    t1 = v.T[:,0]

    abs_max_f1_val = np.max(np.abs(f1))
    abs_max_f1_ix = np.argmax(np.abs(f1)) 
    abs_max_t1_val = np.max(np.abs(t1)) 
    abs_max_t1_ix = np.argmax(np.abs(t1)) 

    neg_peaks_ix = np.argwhere(np.abs(t1)>0.1*abs_max_t1_val)
    if len(neg_peaks_ix)>1:
        neg_first_peak_ix = neg_peaks_ix[-1]

    else:
        neg_first_peak_ix = neg_peaks_ix

    f_neg_peak = (abs_max_f1_ix)*df
    f_neg_bw = np.sum(np.abs(f1)>0.5*abs_max_f1_val)*df
    t_neg_peak = (len(t1) - abs_max_t1_ix)*dt*-1
    t_neg_bw = np.sum(np.abs(t1)>0.5*abs_max_t1_val)*dt

    discard_pos = False
    discard_neg = False
    flip_pos_neg = False
    if l2_norm_neg<discard_thresh*l2_norm_pos:
        discard_neg = True
        f_neg_bw = 0
        t_neg_bw = 0

        
    elif l2_norm_pos<discard_thresh*l2_norm_neg:
        discard_pos = True
        f_pos_bw = 0
        t_pos_bw = 0
        
    if (neg_first_peak_ix>pos_first_peak_ix and not discard_neg) or discard_pos:
        # print('flip_pos_neg = True')
        flip_pos_neg = True
        discard_neg = discard_pos


        f_peak = [f_neg_peak, f_pos_peak]
        f_bw = [f_neg_bw, f_pos_bw]

        t_peak = [t_neg_peak, t_pos_peak]
        t_bw = [t_neg_bw, t_pos_bw]
    else:
        f_peak = [f_pos_peak,f_neg_peak]
        f_bw = [f_pos_bw,f_neg_bw]

        t_peak = [t_pos_peak,t_neg_peak]
        t_bw = [t_pos_bw,t_neg_bw]

    # flags = [flip_pos_neg, discard_neg]

    return [f_peak,f_bw, t_peak,t_bw, flip_pos_neg, discard_neg]


def flip_neg_weights(weights,n_h = 40, dt = 5,dF = 1/6):
    numweights = weights.shape[0]

    mf_peak = np.empty([numweights,2])
    mf_bw = np.empty([numweights,2])
    mt_bw = np.empty([numweights,2])
    mt_peak = np.empty([numweights,2])
    m_pow = np.empty([numweights, n_h])

    flip_pos_neg = np.empty([numweights])
    discard_neg = np.empty([numweights])
    for ii in np.arange(numweights):

        #normalize weight so that all are in same range
        this_weight = weights[ii,:,:]
        this_weight_norm = this_weight/np.max(np.abs(this_weight[:]))
        [mf_peak[ii,:],mf_bw[ii,:], mt_peak[ii,:],mt_bw[ii,:], flip_pos_neg[ii], discard_neg[ii]] = getPeaksAndBWs(this_weight_norm,dt,dF)
        if flip_pos_neg[ii]:

            this_weight = -this_weight
        weights[ii,:,:] = this_weight
    return weights

def quantify_strfs(weights,n_h = 40, dt = 5,dF = 1/6):
 
    numweights = weights.shape[0]

    mf_peak = np.empty([numweights,2])
    mf_bw = np.empty([numweights,2])
    mt_bw = np.empty([numweights,2])
    mt_peak = np.empty([numweights,2])
    m_pow = np.empty([numweights, n_h])

    flip_pos_neg = np.empty([numweights])
    discard_neg = np.empty([numweights])
    # Get measures for real and model data
    for ii in np.arange(numweights):

        #normalize weight so that all are in same range
        this_weight = cp.deepcopy(weights[ii,:,:])
        if np.max(np.abs(this_weight[:]))>0:
            this_weight /=np.max(np.abs(this_weight[:]))


        [mf_peak[ii,:],mf_bw[ii,:], mt_peak[ii,:],mt_bw[ii,:], flip_pos_neg[ii], discard_neg[ii]] = getPeaksAndBWs(this_weight,dt,dF)
        m_pow[ii,:] = np.sum(this_weight**2, axis=0)

    mf_peak_pos = mf_peak[:,0]
    mf_bw_pos = mf_bw[:,0]
    mt_peak_pos = mt_peak[:,0]
    mt_bw_pos = mt_bw[:,0]
    
    mf_peak_neg = mf_peak[np.logical_not(discard_neg),1]
    mf_bw_neg = mf_bw[:,1]
#     mf_bw_neg = mf_bw[np.logical_not(discard_neg),1]
    mt_peak_neg = mt_peak[np.logical_not(discard_neg),1]
#     mt_bw_neg = mt_bw[np.logical_not(discard_neg),1]
    mt_bw_neg = mt_bw[:,1]


    return [mf_peak_pos, mf_peak_neg, mf_bw_pos, mf_bw_neg, mt_peak_pos, mt_peak_neg, mt_bw_pos, mt_bw_neg, m_pow]

def add_mean_ks(this_pd):
    # temp_pd = this_pd.copy()
    colnames = []
    # keys = this_pd.colnames
    # print(this_pd.keys().unique())

    for colname in this_pd.columns:
        # print(colname)
        # print(('peak' in colname and 'pos'in colname) )
        # if 'ks' in colname:

        if 'ks' in colname and ('bw' in colname):# or ('peak' in colname and 'pos' in colname)):
            colnames.append(colname)

    print(colnames)
    this_pd['mean_ks'] = 0
    # tempp = this_pd[colnames]
    n_measures = 0
    for colname in colnames:
        #     print(this_pd[colname])
        this_pd['mean_ks'] += this_pd[colname]
        n_measures += 1
    this_pd['mean_ks'] /= n_measures
    return this_pd

def compare_real_model_distributions(mstrfs, rstrfs, pd_entry):
    [mf_peak_pos, mf_peak_neg, mf_bw_pos, mf_bw_neg, mt_peak_pos, mt_peak_neg, mt_bw_pos, mt_bw_neg, m_pow] = quantify_strfs(mstrfs)
    [rf_peak_pos, rf_peak_neg, rf_bw_pos, rf_bw_neg, rt_peak_pos, rt_peak_neg, rt_bw_pos, rt_bw_neg, r_pow] = quantify_strfs(rstrfs, n_h=38)


    #Exclude any entries where bw=0
    mf_ix = [mf_bw_neg>0] #and [mf_bw_pos>0]
    rf_ix = [rf_bw_neg>0] #and [rf_bw_pos>0]
    mt_ix = [mt_bw_neg>0] #and [mt_bw_pos>0]
    rt_ix = [rt_bw_neg>0] #and [rt_bw_pos>0]

    mf_bw_pos = mf_bw_pos[mf_ix]
    rf_bw_pos = rf_bw_pos[rf_ix]
    mt_bw_pos = mt_bw_pos[mt_ix]
    rt_bw_pos = rt_bw_pos[rt_ix]

    mf_bw_neg = mf_bw_neg[mf_ix]
    rf_bw_neg = rf_bw_neg[rf_ix]
    mt_bw_neg = mt_bw_neg[mt_ix]
    rt_bw_neg = rt_bw_neg[rt_ix]


    ks_t_bw = np.zeros([2])
    ks_f_bw = np.zeros([2])
    ks_t_peak = np.zeros([2])
    ks_f_peak = np.zeros([2])

    [ks_t_bw[0],p] = stats.ks_2samp(mt_bw_pos,rt_bw_pos)
    [ks_t_bw[1],p] = stats.ks_2samp(mt_bw_neg,rt_bw_neg)
    [ks_t_peak[0],p] =stats.ks_2samp((mt_bw_pos-mt_bw_neg)/(mt_bw_pos+mt_bw_neg),(rt_bw_pos-rt_bw_neg)/(rt_bw_pos+rt_bw_neg))  
    [ks_t_peak[1],p] =stats.ks_2samp((mt_bw_pos-mt_bw_neg),(rt_bw_pos-rt_bw_neg))
    # [ks_t_peak[0],p] =stats.ks_2samp((mt_bw_pos-mt_bw_neg),(rt_bw_pos-rt_bw_neg))  
    # [ks_t_peak[1],p] =stats.ks_2samp((mt_bw_pos-mt_bw_neg),(rt_bw_pos-rt_bw_neg))  
    # [ks_t_peak[1],p] =stats.ks_2samp(mt_peak_neg,rt_peak_neg)
    [ks_f_bw[0],p] = stats.ks_2samp(mf_bw_pos,rf_bw_pos)
    [ks_f_bw[1],p] = stats.ks_2samp(mf_bw_neg,rf_bw_neg) 
    # [ks_f_peak[0],p] = stats.ks_2samp(mf_peak_pos,rf_peak_pos)
    # [ks_f_peak[1],p] = stats.ks_2samp(mf_peak_neg,rf_peak_neg) 
    [ks_f_peak[0],p] =stats.ks_2samp((mf_bw_pos-mf_bw_neg)/(mf_bw_pos+mf_bw_neg),(rf_bw_pos-rf_bw_neg)/(rf_bw_pos+rf_bw_neg))  
    [ks_f_peak[1],p] =stats.ks_2samp((mf_bw_pos-mf_bw_neg),(rf_bw_pos-rf_bw_neg))  
    # [ks_f_peak[0],p] =stats.ks_2samp((mf_bw_pos-mf_bw_neg),(rf_bw_pos-rf_bw_neg))  
    # [ks_f_peak[1],p] =stats.ks_2samp((mf_bw_pos-mf_bw_neg),(rf_bw_pos-rf_bw_neg))  
    ks_t_peak[np.isnan(ks_t_peak)] = 1
    ks_t_bw[np.isnan(ks_t_bw)] = 1
    ks_f_bw[np.isnan(ks_f_bw)] = 1
    ks_f_peak[np.isnan(ks_f_peak)] = 1
    
    if pd_entry is not None:
        pd_entry['f_peak_pos'] = mf_peak_pos
        pd_entry['f_peak_neg'] = mf_peak_neg
        pd_entry['f_bw_pos'] = mf_bw_pos
        pd_entry['f_bw_neg'] = mf_bw_neg
        pd_entry['t_peak_pos'] = mt_peak_pos
        pd_entry['t_peak_neg'] = mt_peak_neg
        pd_entry['t_bw_pos'] = mt_bw_pos
        pd_entry['t_bw_neg'] = mt_bw_neg
        pd_entry['t_pow'] = m_pow
        pd_entry['ks_f_peak_pos'] = ks_f_peak[0]
        pd_entry['ks_f_peak_neg'] = ks_f_peak[1]
        pd_entry['ks_t_peak_pos'] = ks_t_peak[0]
        pd_entry['ks_t_peak_neg'] = ks_t_peak[1]
        pd_entry['ks_t_bw_pos'] = ks_t_bw[0]
        pd_entry['ks_t_bw_neg'] = ks_t_bw[1]
        pd_entry['ks_f_bw_pos'] = ks_f_bw[0]
        pd_entry['ks_f_bw_neg'] = ks_f_bw[1]

        return pd_entry


#     print(ks_t_bw)
    return [ks_t_peak, ks_t_bw, ks_f_bw]


def compare_real_model_populations(this_pd, rstrfs, display=1, keep_prop=0.01):

    n_h = 40
    num_freq = 32

    out_pd = this_pd.copy()
    ii= 0
    for entry_loc,this_entry in this_pd.iterrows():

        pth = this_entry['results_path']

        pred_net = pkl.load(open(pth, 'rb'))
        if not isinstance(pred_net, dict):
            network_params  = pred_net.network_params
            cost_history = pred_net.cost_history
        else:
            network_params = pred_net['network_params']
            cost_history = pred_net['cost_history']
            
        weights = network_params[0].T
        l2_norm = np.sum(weights**2,axis=1)
        keep_ix = l2_norm>keep_prop*max(l2_norm)
        mstrfs = weights[keep_ix,:]
        num_mstrfs = mstrfs.shape[0]
        mstrfs =np.reshape(mstrfs,[num_mstrfs, n_h, num_freq])
        mstrfs = np.rollaxis(mstrfs,2,1)

        if ii == 0:
            temp = compare_real_model_distributions(mstrfs, rstrfs, this_entry)
            for cname in temp.index:
                if cname not in out_pd.columns:
                    temp[cname] = None
                else:
                    temp = temp.drop(cname)
            out_pd = out_pd.assign(**temp)
        
        this_entry = compare_real_model_distributions(mstrfs, rstrfs, this_entry)

        out_pd.loc[entry_loc,this_entry.index.tolist()] = this_entry

        ii+=1
    return pd.DataFrame(out_pd)
