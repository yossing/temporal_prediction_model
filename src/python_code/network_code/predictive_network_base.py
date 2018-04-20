import os    
import copy
import pickle as pkl
from imp import reload
import numpy as np
import theano
import lasagne
from visualisation.network_visualisation import plot_loss 
from network_code import optimizer_lasagne
reload(optimizer_lasagne)

#-------------------------------------------------------------------------------
#----------------------------------CONSTANTS------------------------------------
#-------------------------------------------------------------------------------

FLOATX = 'float32' # needed to use the GPU
GRAD_CLIP = 100
# DEFAULT_N_HIDDEN = 100
# DEFAULT_N_LAYERS = 1
# DEFAULT_BATCH_PERC = 0.05

NON_LINEARITIES = {'sigmoid':lasagne.nonlinearities.sigmoid,
                   'tanh':lasagne.nonlinearities.tanh,
                   'rectify':lasagne.nonlinearities.rectify,
                   'relu':lasagne.nonlinearities.rectify,
                   'leaky_rectify':lasagne.nonlinearities.leaky_rectify,
                   'softplus':lasagne.nonlinearities.softplus,
                   'linear':None}

UPDATE_FUNCS = {'sgd': lasagne.updates.sgd,
                'momentum': lasagne.updates.momentum,
                'nesterov_momentum': lasagne.updates.nesterov_momentum,
                'adagrad': lasagne.updates.adagrad,
                'rmsprop': lasagne.updates.rmsprop,
                'adadelta': lasagne.updates.adadelta,
                'adam': lasagne.updates.adam}
REGULARIZATION = {'l1':lasagne.regularization.l1,
                  'l2':lasagne.regularization.l2}

class PredictiveNetwork(object):     

    """
    Predictive coding network. 
    Network and cost settings as well as other 
    class attributes can be set by passing in **kwargs 
    when instantiating.
    """

    def __init__(self, **kwargs):

        self._init_default_attributes()
      
        for key, value in kwargs.items():
            if key in self.network_settings:
                self.network_settings[key] = value
            elif key in self.cost_settings:
                self.cost_settings[key] = value
            elif key in self.cost_history:
                self.cost_history[key] = value
            elif key in self.input_settings:
                self.input_settings[key] = value
                # print('recieved: ' + key + ' in input settings, overwriting default value with: ' + str(value))
            #TODO:check that this works!
            elif hasattr(self, key):
                # print(key)
                # print(value)
                setattr(self, key, value)
                
        # if self.network is None:
        # elif ((self.train_fn is None) or (self.val_fn is None)):
        #     self._init_cost_funcs()

#-------------------------------------------------------------------------------
#---------------------------Initialization functions----------------------------
#-------------------------------------------------------------------------------
    def _init_default_attributes(self):
        self.network_settings = {'nonlinearity': 'sigmoid',
                                 'is_recurrent': False,
                                 'model': None,#'fully_connected_nn',
                                 'num_hidden_units': None,
                                 'num_layers': None,
                                 'dropout': 0}

        self.cost_settings = {'update_func': 'adam',
                              'regularization': 'l1',
                              'reg_factor': 0,
                              'learning_rate': 0.01,
                              'act_reg': None,
                              'act_reg_factor': None, 
                              'elastic_alpha': None, 
                              'output_distribution': None}
        self.cost_history = {'train_costs': [],
                             'val_costs': [],
                             'final_train_cost': 0,
                             'final_val_cost': 0}

        self.input_settings = {'RF_size': 20,
                               'filter_type': 2,
                               'noise_ratio': 0,
                               'input_noise_ratio': 0,
                               'data_path': '', 
                               'post_dict': False, 
                               'norm_type': 0,
                               't_past':7, 
                               't_future':1}

        #Theano specific variables
        # self.theano_vars = {'network':None,
        #                     '_input_var':None,
        #                     '_target_var':None,
        #                     'train_fn':None,
        #                     'val_fn':None}
        self.network = None
        self._input_var = None
        self._target_var = None
        self.train_fn = None
        self.val_fn = None

        self.network_param_values = None
        self.initial_param_values = None
        self.save_path = '' #TODO perhaps this doesn't belong here
        
    
    # Because theano vars are dependant on CudNN or similar variables, it is not a 
    # good idea to save them together with the rest of the model as they won't load 
    # on computers with different backends. To get around this, define functions to 
    # construct and delete theano dependant vars.
    

    def init_theano_vars(self):
        """
        NB: requires that all necessary class atributes are already initialised
        correctly. Depending on the subclass, these will vary.

        Can call this either from __init__ when instantiaiting the object for
        the first time or when loading from pickle and need to recreate the theano vars 
        """
        self._init_network()
        self._init_cost_funcs()

        if self.network_param_values is not None:
            lasagne.layers.set_all_param_values(self.network, 
                                                self.network_param_values)

        return 

    def erase_theano_vars(self):
        """
        Set all Theano dependant variables to None
        """
        self.network = None
        self._input_var = None
        self._target_var = None
        self.train_fn = None
        self.val_fn = None
        self.test_prediction = None
        self.precision = None
        return 


    def _init_network(self, *args, **kwargs):
        """
        This is an abstract method that must be implemented in the child class
        """
        raise NotImplementedError()    

    def _init_cost_funcs(self):
        network = self.network
        reg_factor = self.cost_settings['reg_factor']
        regularization = self.cost_settings['regularization']
        update_func = self.cost_settings['update_func']

        prediction = lasagne.layers.get_output(network, deterministic=False)
        if 'output_distribution' in self.cost_settings:
            if self.cost_settings['output_distribution'] is not None: 
                if self.cost_settings['output_distribution'] == 'independent_unimodal_gaussian':
                    mu = prediction[:,:self.n_output_units,:]
                    precision = prediction[:,self.n_output_units:,:]
                    sq_error = lasagne.objectives.squared_error(mu, self._target_var)
                    cost = (np.exp(precision)*sq_error) - precision
                    cost = cost.mean() 
                elif self.cost_settings['output_distribution'] == 'input_independent_unimodal_gaussian':
                    mu = prediction
                    self.precision = theano.shared(np.zeros(self.n_output_units).astype('float32'))
                    sq_error = lasagne.objectives.squared_error(mu, self._target_var)
                    cost = (np.exp(self.precision)*sq_error) - self.precision
                    cost = cost.mean()     
                    # Retrieve all trainable parameters in network
                    params = lasagne.layers.get_all_params(network, trainable=True)
                    params.append(self.precision)

            else:
                cost = lasagne.objectives.squared_error(prediction, self._target_var)
                # cost = cost.sum(axis=2) #sum across space
                # cost = cost.sum(axis=1) #sum across time

                # cost = cost.mean(axis=2) #take the mean across space
                # cost = cost.mean(axis=1) #take the mean across time
                cost = cost.mean() #take the mean across training examples

        if regularization == 'l1' or regularization == 'l2':
            reg_penalty = reg_factor * lasagne.regularization.regularize_network_params(network, REGULARIZATION[regularization])
        else:
            print("Unrecognised regularisation type. Setting to None!")

        cost = cost + reg_penalty
        

        act_reg_factor = self.cost_settings['act_reg_factor']
        act_reg = self.cost_settings['act_reg']
        if act_reg is not None:
            all_layers = lasagne.layers.get_all_layers(network)
            #get rid of inupt and output layers
            hidden_layers = all_layers[1:-1]
            hu_activities = lasagne.layers.get_output(hidden_layers)
            if act_reg =='l2' or act_reg=='l1': 
                act_reg_penalty = act_reg_factor*lasagne.regularization.apply_penalty(hu_activities, REGULARIZATION[act_reg])    
            cost = cost + act_reg_penalty
        
        params = lasagne.layers.get_all_params(network, trainable=True)
        
        # params = lasagne.layers.get_all_params(network)


        u_func = UPDATE_FUNCS[update_func]
        if (update_func == 'sgd' or update_func == 'momentum' 
                or update_func == 'nesterov_momentum'):
            updates = u_func(cost, params, self.network_settings['learning_rate'])
        else:
            updates = u_func(cost, params)

        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        self.test_prediction = theano.function([self._input_var], test_prediction)

        test_cost = lasagne.objectives.squared_error(test_prediction, self._target_var)
        # test_cost = test_cost.sum(axis=1)
        
        test_cost = test_cost.mean()
        self.train_fn = theano.function([self._input_var, self._target_var], cost, 
                                        updates=updates, allow_input_downcast=True)
        self.val_fn = theano.function([self._input_var, self._target_var], cost,
                                      allow_input_downcast=True)
        return self.train_fn, self.val_fn   

#-------------------------------------------------------------------------------
#-------------------------------Public functions--------------------------------
#-------------------------------------------------------------------------------

    def train_network(self, X_train, y_train, X_val= None, y_val = None, 
                      num_epochs = 100, show_graph = False, max_epochs = None):

        if max_epochs is not None:
            n_epochs_run = len(self.cost_history['train_costs'])
            n_to_run = max_epochs - n_epochs_run
            if n_to_run < num_epochs:
                print('Already run %i' %n_epochs_run + ' epochs out of %i' %max_epochs)
                print('Only running %i' %n_to_run + ' epochs instead of %i' %num_epochs)
                num_epochs = n_to_run
                
        if num_epochs>0:

            optimizer = optimizer_lasagne.Optimizer(self.network, 
                                                    self.train_fn, 
                                                    val_fn=self.val_fn,
                                                    verbose=1, 
                                                    batch_size=None)

            print("Starting training ...")

            network_param_values, cost_history = optimizer.optimize(X_train, y_train, 
                                                              X_val=X_val, y_val=y_val, 
                                                              num_epochs=num_epochs,
                                                              )

            self.network_param_values = network_param_values
            self.cost_history['train_costs'].extend(cost_history['train_costs'])
            self.cost_history['val_costs'].extend(cost_history['val_costs'])
            if self.cost_history['val_costs'] !=[]:
                self.cost_history['final_val_cost'] = self.cost_history['val_costs'][-1]
                self.cost_history['final_train_cost'] = self.cost_history['train_costs'][-1]
            if show_graph:
                plot_loss(self.cost_history['train_costs'], 
                              val_losses=self.cost_history['val_costs'])
        return 



    def get_network_results_and_settings(self, use_nans=False):
        '''Get parameters and settings of network'''
        d = {}
        # d['network'] = self.network
        if self.network is not None:
            d['network_param_values'] = lasagne.layers.get_all_param_values(self.network)
        else:
            d['network_param_values'] = self.network_param_values
        d['network_settings'] = self.network_settings
        d['cost_settings'] = self.cost_settings
        d['input_settings'] = self.input_settings
        d['cost_history'] = self.cost_history
        d['save_path'] = self.save_path

        #Would want to replace None with np.nan if saving to .matfile
        if use_nans:
            for kk,vv in d.items():
                if isinstance(vv, dict):
                    for k,v in vv.items():
                        if v is None:
                            vv[k] = np.nan
        return d


    def to_pickle(self, save_path=None):
        '''Pickle the network object'''
        if save_path is not None:
            self.save_path = save_path
        save_path = os.path.expanduser(self.save_path)

        if not os.path.exists(os.path.dirname(os.path.abspath(save_path))):
            os.makedirs(os.path.dirname(os.path.abspath(save_path)))

        #ensure the network_param_values are up to date:
        self.network_param_values = lasagne.layers.get_all_param_values(self.network)
        #first eliminate any theano dependant vars
        #but if we are in mid-training and just want to save a copy at an intermediate
        #stage, then don't want to lose update parameters (such as momentum etc.)
        #to avoid this, make a copy, delete theano vars and save the copy
        obj_to_save = copy.copy(self)
        obj_to_save.erase_theano_vars()
        pkl.dump(obj_to_save, open(save_path, 'wb'))
        # #instead of saving the pn object which is dependant on theano variables, 
        # # rather save the network settings etc
        # pkl.dump(self.get_network_results_and_settings(use_nans=False), open(save_path, 'wb'))
        return


#function rather than class method
def from_pickle(load_path):
    load_path = os.path.expanduser(load_path)
    loaded_pn = pkl.load(open(load_path, 'rb'))
    loaded_pn.init_theano_vars()
    return loaded_pn

