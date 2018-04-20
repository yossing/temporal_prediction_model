All of the custom code used to in the paper is presented here. 

The code to preprocess  auditory data and analyse visual results is written in MATLAB. 
All other code is written in Python. 

# Setup instructions
First clone the repository or download the ./src directory

## To use MATLAB code:
No setup required. 

## To use Python code:

cd into python_code directory

Install python dependencies using pip:

	pip install -r virtual_requirements.txt

Install latest versions of Theano and Lasagne:

	pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
	pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

It is strongly recommended to train networks using a GPU.
To setup Theano to use the GPU follow the instructions on Theano's website: http://deeplearning.net/software/theano/tutorial/using_gpu.html

# To train networks

## Preprocessing visual data

Follow the example in create_visual_dataset.ipynb

## Preprocessing auditory data

Follow the instructions in [/matlab_code/auditory_preprocessing/README.md](matlab_code/auditory_preprocessing/README.md)

## Training networks

Follow the examples given in train_networks.ipynb
If you would like to perform a grid search over hyperparameters, try out the package in https://github.com/yossing/distributed_grid_search/

# Analysing Receptive Fields (RFs)

## Visual RFs
First fit Gabors to the visual RFs (input weights). This can be done by running [matlab_code/visual_analysis/fitGabors_best_t_step.m](matlab_code/visual_analysis/fitGabors.m)

To perform the analysis and make the plots presented in the paper, use the plotting functions in [matlab_code/visual_analysis/](matlab_code/visual_analysis/)

## Auditory RFs
To perform the analysis and make the plots presented in the paper, follow the examples in [plot_auditory_figures.ipynb](python_code/plot_auditory_figures.ipynb)
