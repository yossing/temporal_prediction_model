Code to preprocess sounds and convert them to auditory stimuli which are in a format that can be passed direcctly to the neural network. 

Start at soundPreprocessingMain.m.
	Here is where you input the paths to the folders containing sound files, and to the directories where you would like to save the cochleagrams and the finalised datasets to. Once these are defined, you can call makeAndSaveSoundStimuli. 
	You can use this script to preprocess multiple datasets sequentially by defining the paths and then calling makeAndSaveSoundStimuli. 

	See Notes at the bottom for further info.

In makeAndSaveSoundStimuli:
	This function handles the details of dataste creation. It first fetches the cochleagrams and then calls finaliseAuditoryDataset which selects portions of the cochleagrams from different soundfiles and puts them into the finalised format so that they can be input straight into the neural network. The dataset can then be further divided into 'subrefs' (batches of training examples) as is required by the Sum Of Functions optimizer.

	Functions called are: 
	-fetchTensorizedCochleagrams
	-finaliseAuditoryDataset
	-createSoundSubrefs (if create_subrefs option is true)
In FetchTensorizedCochleagrams:
	This will either create or load previously created cochleagrams
	If overwrite cochleagrams option is true or if a set of tensorized cochleagrams have not previously been created int he specified save directory, then these will be created by calling makeTensorizedCochleagrams. Otherwise the cochleagrams and the settings used to make them will be loaded.
In makeTensorizedCochleagrams:
	The default cochleagram settings are defined. These can be overwritten if a struct called settings is passed into the function (as a varargin).
	All of the sound files in the specified folder (up to a specified maximum number of files) are then loaded in in batches. Each sound file is converted to a cochleagram which is subsequentoly tensorized with n_h history bins. Time bins which contain blank spaces (when there is 0 variance across all frequency bins) are removed. The cochleagram is then added to a cell array containing all of the previously generated cochleagrams. The entries into the array are saved incrementally to a matfile after each batch of files is processed. This allows many files to be processed without running into memory issues. 

	Functions called include:
	-cochleagram (or power_cochleagram, depending on the settings). 
	-tensorize
	These are both acquired directly from BenWare. cochleagram and power_cochleagram both depend on melbank.m. tensorize is standalone. 
In finaliseAuditoryDataset:
	An cell array of tensorized cochleagrams is passed in. Each entry contains the cochleagram from a differrent sound file. These are then parsed, with a specified number of training examples being taken from each entry and added to the training and test datasets. This is done until the dataset reaches a specified number of examples or until all of the entries of the cell array have been parsed. 
	Traning and test examples are selected and added to the dataset in concatenateSoundArrays. By default, 20% of examples go to the training set and the remainder to the test set. 

	Once the training and test sets have been comiled, they are normalised and may be passed through a hill function if this option has been selected. 
	Before passing through the hill function, each frequency band is divided by its median value (actually by 0.2*median).

	The training and test datasets are then normalised by subtracting the mean of the training set and dividing by its standard deviation. Once again, this can be applied per frequency or not by selecting the norm_type. 

Notes:
	All (or most) functions accept varargins which are name-value pairs that can be passed in from calling functions and override the default options when they exist. Varargin can be passed into any calling function and are passed through to subroutines (without affecting calling functions) until they appply. 

	There are three key file directories that are specified: 
	-sound_file_dir: This contains all of the sound files you want to preprocess. Do not keep things in sub-directories.
	-cochleagram_save_dir: If this is empty, this is where cochleagrams will be saved to a matfile named all_concattens.mat. This is a struct with entries Fss, which lists the sampling frequencies of all the sounds included. The actual tensorized cochleagrams are in an entry called all_concattens. So all_concattens.all_concattens contains the tensorized cochleagrams. This is probably a bit confusing, but you can change the name settings in makeTensorizedCochleagrams.
	-final_save_dir. This is where the final formatted and normalised dataset will be saved.





	
