'''
Predictive network subclasses. All inherit from predictive_network_base.PredictiveNetwork 
and extend to specific network configurations. 
Most of the methods are implemented in the base class. 
'''

#-------------------------------------------------------------------------------
#------------------------------------Imports------------------------------------
#-------------------------------------------------------------------------------
import os
from imp import reload
from network_code import predictive_network_base as pn_base
from network_code.predictive_network_base import from_pickle
import theano.tensor as T
import lasagne
reload(pn_base)
#-------------------------------------------------------------------------------
#----------------------------------CONSTANTS------------------------------------
#-------------------------------------------------------------------------------

FLOATX = 'float32' # needed to use the GPU
GRAD_CLIP = 100
DEFAULT_N_HIDDEN = 100
DEFAULT_N_LAYERS = 1
DEFAULT_BATCH_PERC = 0.05

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


#-------------------------------------------------------------------------------
#---------------------------Fully connected network-----------------------------
#-------------------------------------------------------------------------------

class PredictiveFCN(pn_base.PredictiveNetwork):  

    """
    Fully connected Predictive Network. 
    Defines configuration specific init methods. 
    All public methods remain in parent (PredictiveNetwork) class
    """

    def __init__(self, in_shape, out_shape, **kwargs):
        super().__init__(**kwargs)    
            
        # self._init_network(in_shape, out_shape)
        # self._init_cost_funcs()
        if in_shape is not None and out_shape is not None:
            self.in_shape = in_shape
            self.out_shape = out_shape

        self.init_theano_vars()

        if self.network_param_values is not None:
            lasagne.layers.set_all_param_values(self.network, 
                                                self.network_param_values)

    def _init_network(self):

        in_num_feats = self.in_shape[-1]
        out_num_feats = self.out_shape[-1]

          # prepare Theano variables for inputs and targets
        self._input_var = T.matrix('inputs', dtype=FLOATX)
        self._target_var = T.matrix('targets', dtype=FLOATX)
        self.network = self._build_fully_connected_network(in_num_feats, out_num_feats)
        self.network_settings['is_recurrent'] = False
        self.initial_param_values = lasagne.layers.get_all_param_values(self.network)

        return self.network

    def _build_fully_connected_network(self, in_num_feats, out_num_feats):
        if self.network_settings['nonlinearity'] == 'scaled_tanh':
            rho1 = 1.7159
            rho2 = 2/3
            nonlinearity = lasagne.nonlinearities.ScaledTanH(scale_in=rho2, scale_out=rho1)
        else:
            nonlinearity = NON_LINEARITIES[self.network_settings['nonlinearity']]
        
        network = lasagne.layers.InputLayer(shape=(None, in_num_feats), input_var=self._input_var)
        #batchsize, _ = network._input_var.shape
        #network = ReshapeLayer(network,shape=(-1,num_hidden_units))
        for _ in range(self.network_settings['num_layers']):
            network = lasagne.layers.DenseLayer(network,
                                 num_units=self.network_settings['num_hidden_units'],
                                 nonlinearity=nonlinearity)
        network = lasagne.layers.DenseLayer(network, num_units=out_num_feats, nonlinearity=None)
        #network = ReshapeLayer(network,shape=(batchsize, out_num_feats))
        return network        

class PredictiveConv1DNetwork(pn_base.PredictiveNetwork):  

    """
    PredictiveNetowrk for performing convoluitons along one dimension (time). 
    Defines configuration specific init methods. 
    All public methods remain in parent (PredictiveNetwork) class
    """

    def __init__(self, in_shape=None, out_shape=None, **kwargs):
        '''
        @in_shape should be in the form [n_batches,x,t]
        @out_shape should be in the form [n_batches,x,t'] -> t' will differ depending on 
        whether we use a full/valid/same conv
        @**kwargs: All other customizable parameters should be passed in through kwargs
        
        '''

        #Call the parent's __init__ function 
        # print(kwargs)
        # print(**kwargs)
        super().__init__(**kwargs)   
        self.conv_settings = {'t_filter_length':7,
                              'num_filters':400,
                              't_predict_length': 1,
                              'pad':'valid',
                              'stride':1}
        # print(kwargs)
        # print(**kwargs)
        for key, value in kwargs.items():
            if key in self.conv_settings:
                self.conv_settings[key] = value                

        if in_shape is not None and out_shape is not None:
            self.in_shape = in_shape
            self.out_shape = out_shape

        self.init_theano_vars()

        if self.network_param_values is not None:
            lasagne.layers.set_all_param_values(self.network, 
                                                self.network_param_values)
        return

    def _init_network(self):

        in_shape = self.in_shape
        out_shape = self.out_shape
        in_num_feats = in_shape[1:]
        out_num_feats = out_shape[1:]

        self.n_output_units = out_num_feats[0] 

        #tuples are immutable, so direct assignment won't work. Temporarily assign to list
        in_lst = list(in_num_feats)
        out_lst = list(out_num_feats)
        in_lst[-1] = None
        out_lst[-1] = None
        if 'output_distribution' in self.cost_settings: 
            if self.cost_settings['output_distribution'] == 'independent_unimodal_gaussian':
                out_lst[0] = 2*out_lst[0]
        in_num_feats = tuple(in_lst)
        out_num_feats = tuple(out_lst)

        # print(in_num_feats)
        # print(out_num_feats)
        # in_seq_length = 1
        # out_seq_length = 1

      # prepare Theano variables for inputs and targets
        #Size of each input batch is nxqxt
        self._input_var = T.tensor3('inputs', dtype=FLOATX)
        #Size of each target batch is nxqx[t']
        self._target_var = T.tensor3('targets', dtype=FLOATX)

        self.network = self._build_network(in_num_feats, out_num_feats)

        #Also initilialize cost functions
        # self._init_cost_funcs()
        if self.initial_param_values is None: 
            self.initial_param_values = lasagne.layers.get_all_param_values(self.network)

        return self.network

    def _build_network(self, in_num_feats, out_num_feats):

        if self.network_settings['nonlinearity'] == 'scaled_tanh':
            rho1 = 1.7159
            rho2 = 2/3
            nonlinearity = lasagne.nonlinearities.ScaledTanH(scale_in=rho2, scale_out=rho1)
        else:
            nonlinearity = NON_LINEARITIES[self.network_settings['nonlinearity']]
        in_shape = tuple([None])+tuple(in_num_feats)
        # print("in_shape")
        # print(in_shape)
        network = lasagne.layers.InputLayer(shape=in_shape, input_var=self._input_var)

        #batchsize, _ = network._input_var.shape
        #network = ReshapeLayer(network,shape=(-1,num_hidden_units))
        # for ii in range(2):
        if self.network_settings['nonlinearity'] == 'rectify' or self.network_settings['nonlinearity'] == 'relu' or self.network_settings['nonlinearity'] == 'elu' or self.network_settings['nonlinearity'] == 'softplus':
            network = lasagne.layers.Conv1DLayer(network,
                                                 num_filters=self.conv_settings['num_filters'],
                                                 filter_size=(self.conv_settings['t_filter_length']),
                                                 stride=1,
                                                 pad='valid',
                                                 nonlinearity=nonlinearity, 
                                                 flip_filters=False,
                                                 W=lasagne.init.GlorotUniform('relu'), b=lasagne.init.Constant(1.0))
                        # network = ReshapeLayer(network,shape=(-1,num_hidden_units))
        else:   
            network = lasagne.layers.Conv1DLayer(network,
                                                 num_filters=self.conv_settings['num_filters'],
                                                 filter_size=(self.conv_settings['t_filter_length']),
                                                 stride=1,
                                                 pad='valid',
                                                 nonlinearity=nonlinearity, 
                                                 flip_filters=False)
                # network = ReshapeLayer(network,shape=(-1,num_hidden_units))
        if 'dropout' in self.network_settings:
            dropout = self.network_settings['dropout'] 
            if dropout > 0:
                print('adding drouput layer')
                network = lasagne.layers.DropoutLayer(network, p=dropout, rescale=True)

        #The output layer is also convolutional, but in this case, it is qx1
        network = lasagne.layers.Conv1DLayer(network,
                                             num_filters=out_num_feats[0],
                                             filter_size=(1), #(self.conv_settings['t_predict_length']),
                                             stride=1,
                                             pad='valid',
                                             nonlinearity=None, 
                                             flip_filters=False)
        #network = ReshapeLayer(network,shape=(batchsize, out_num_feats))
        return network        

    def get_network_results_and_settings(self, use_nans=False):
        '''
        call parent class's implementation and add conv_settings to
        dictionary
        '''
        res_dict = super().get_network_results_and_settings(use_nans)
        res_dict['conv_settings'] = self.conv_settings
        return res_dict