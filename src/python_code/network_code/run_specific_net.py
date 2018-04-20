from imp import reload
from network_code import predictive_network_subclasses as pn
from network_code import data_handling as dat
reload(dat)
reload(pn)

def run_network():
	data_path = '/full/path/to/preprocessed/training/data/.pkl'
	save_path = '/full/path/to/save/location/.pkl'

	noise_ratio = 0

	[X_train, y_train, X_val, y_val] = dat.load_1d_conv_vis_data(data_path, norm_type=1)

	network_settings = {}

	network_settings['noise_ratio'] = noise_ratio
	network_settings['nonlinearity'] = 'sigmoid'
	network_settings['num_filters'] = 400
	network_settings['regularization'] = 'l1'
	network_settings['reg_factor'] = 10**-6
	network_settings['update_func'] = 'adam'
	network_settings['save_path'] = save_path

	ff_net = pn.PredictiveConv1DNetwork(X_train.shape, y_train.shape, **network_settings)

	for _ in range(10):
	    ff_net.train_network(X_train, y_train, X_val, y_val, num_epochs=100, show_graph=False)
	    ff_net.to_pickle()

def main():
    run_network()

if __name__ == "__main__":
    main()