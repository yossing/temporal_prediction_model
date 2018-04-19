import numpy as np
import lasagne
import time
FLOATX='float32' # needed to use the GPU

DEFAULT_BATCH_PERC = 0.01
DEFAULT_BATCH_SIZE = 7000

class Optimizer(object):

    def __init__(self, network, train_fn, val_fn=None,verbose=1, batch_size=None):

        self.train_fn = train_fn
        self.val_fn = val_fn
        self.network = network
        #self.batch_size = predictive_network.batch_size
        if batch_size is None:
            self.batch_size = DEFAULT_BATCH_SIZE
        else:
            self.batch_size = batch_size
        self.verbose = verbose
        self._orig_start_time = None
        self._num_epochs_run = 0
        self.cost_history = {'train_costs': [],
                             'val_costs': [],
                             'final_train_cost': 0,
                             'final_val_cost': 0}


    #-------------------------------------------------------------------------------
    #-----------------------------Update functions----------------------------------
    #-------------------------------------------------------------------------------

    def run_all_epochs(self, X_train, y_train, X_val=None, y_val=None, num_epochs=100):
        # train_costs = []
        # val_costs = []
        
        if num_epochs is not None:
            self.run_n_epochs(self, X_train, y_train, X_val=X_val, y_val=y_val, num_epochs=num_epochs)
        # else:
        #     self._run_epochs_till_converge(self, X_train, y_train, X_val=None, y_val=None)

        return 



    def run_n_epochs(self, X_train, y_train, X_val, y_val, num_epochs):

        for epoch in range(num_epochs):
            start_time = time.time()

            if X_val is not None:
                val_err = self._run_an_epoch(X_val, y_val, self.val_fn)
                self.cost_history['val_costs'].append(val_err)
            else:
                val_err = None

            train_err = self._run_an_epoch(X_train, y_train, self.train_fn)
            self.cost_history['train_costs'].append(train_err)

            if self.verbose: 
                print_progress(epoch, num_epochs, train_err, start_time,
                               val_err=val_err, 
                               final_epoch=False, 
                               print_period=np.minimum(int(num_epochs/10),100))

            if epoch == num_epochs - 1:
                print_progress(epoch, num_epochs, train_err, self._orig_start_time,
                               val_err=val_err,
                               final_epoch=True,
                               print_period=1)


    def _run_an_epoch(self, X, y, cost_fn):
        # print(len(y))
        num_examples = X.shape[0]
        # print(num_examples)
        current_err = 0
        batches = 0
        batch_perc = DEFAULT_BATCH_PERC

        if self.batch_size is not None:
            if num_examples < (5*self.batch_size):
                batch_size = int(batch_perc*num_examples)
            else:
                batch_size = self.batch_size
        else:
            batch_size = int(batch_perc*num_examples)
        # print(batch_size)
        for batch in self._iterate_minibatches(X, y, 
                                               batch_size, 
                                               shuffle=True):
            if y is not None:
                inputs, targets = batch
                current_err += cost_fn(inputs, targets)
            else:
                inputs = batch
                current_err += cost_fn(inputs)
            batches += 1
        # print(batches)
        current_err /= batches
        # current_err /=num_examples
        self._num_epochs_run += 1
        return current_err


    def _iterate_minibatches(self,inputs, targets, batchsize, shuffle=True):

        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):

            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize,1)
            # print(excerpt)
            if targets is not None:
                assert(len(inputs) == len(targets))
                yield inputs[excerpt, ...], targets[excerpt, ...]
            else:
                yield inputs[excerpt, ...]


    #-------------------------------------------------------------------------------
    #-----------------------------Public functions----------------------------------
    #-------------------------------------------------------------------------------
    def optimize(self, X_train, y_train, X_val=None, y_val=None, num_epochs = 100):
        self._orig_start_time = time.time()
        self.run_n_epochs(X_train, y_train, X_val, y_val, num_epochs)
        self.cost_history['final_train_cost'] = self.cost_history['train_costs'][-1]
        if self.cost_history['val_costs']:
            self.cost_history['final_val_cost'] = self.cost_history['val_costs'][-1]

        network_param_values = lasagne.layers.get_all_param_values(self.network)
        cost_history = self.cost_history
        self.reset_cost_history()
        return network_param_values, cost_history

    def reset_cost_history(self):
        """
        public function to reset the cost histories to empty arrays
        """
        self.cost_history = {'train_costs': [],
                             'val_costs': [],
                             'final_train_cost': 0,
                             'final_val_cost': 0}



#-------------------------------------------------------------------------------
#-----------------------------Output functions----------------------------------
#-------------------------------------------------------------------------------
def print_progress(epoch, num_epochs, train_err, start_time,
                   val_err=None, 
                   final_epoch=False, 
                   print_period=10):
    min_period = 10
    if epoch < min_period or epoch % print_period == 0:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training cost:\t\t{:.6f}".format(train_err))
        if val_err is not None:
            print("  validation cost:\t\t{:.6f}".format(val_err))

    if final_epoch:
        print("Total time took {:.3f}s".format(time.time() - start_time))
    return



