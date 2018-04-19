function [concattrain,concattest] = finaliseAuditoryDataset(tensorized_cochleagrams, varargin)
%Put the cochleagrams into a format that is usable as input to neural nets. 
%Also, divide into training and test sets while doing this
%---normtype: 0->normalize by mu and std of dataset, 1->per freq normalization
nFiles = size(tensorized_cochleagrams,1);
rng(42);

concatTrainArray = [];
concatTestArray = [];

opts.verbose = true;

opts.num_clip_select = 20000;
opts.min_clips_to_add = 20;
opts.ds_size = 500000;

opts.normtype = 0; %0->normalize by mu and std of dataset, 1->per freq norm

opts.applyHill = true;

if length(varargin)==1&&isempty(varargin{1}),varargin = {};end;
[opts, varargin] = vl_argparse(opts, varargin) ;

batch_size = 1;
total_batches = floor(nFiles/batch_size);
batch_idx = randperm(total_batches);
done = false;
batch_count = 1;

tic;
try
    while (size(concatTrainArray,2)<opts.ds_size) && ~done

        if opts.verbose
            display(['Completed ',num2str(100*size(concatTrainArray,2)/opts.ds_size),'% of preprocessing']);
            toc;
        end

        %select a bunch of files:
        batch = batch_idx(batch_count); %allows us to randomise which files are included in dataset
        start_file_idx = 1+(batch-1)*batch_size;
        end_file_idx = start_file_idx+batch_size-1;
        if end_file_idx > nFiles
            end_file_idx = nFiles;
            done = true;
        end
        if start_file_idx>end_file_idx
            break;
        end
            
        these_file_idx = start_file_idx:1:end_file_idx;	
        these_tens = tensorized_cochleagrams(1,start_file_idx:end_file_idx);

        for ii = 1:length(these_tens)
            this_tens = these_tens{ii};
            [nf,nh,nt] = size(this_tens);
            this_tens = reshape(this_tens,nf*nh,nt);
            % all_processed_clips{start_file_idx+ii-1} = this_processed_clip;
            if size(this_tens,2)>=opts.min_clips_to_add
            [concatTestArray, concatTrainArray] = concatenateSoundArrays(this_tens,...
                                                  concatTrainArray,concatTestArray,...
                                                  opts.num_clip_select);    
            end
        end
        batch_count = batch_count+1;
        if batch_count>total_batches
            done = true;
        end
          

    end
	
	% permutation = randperm(size(all_processed_clips, 2));
	% concat_test = all_processed_clips(permutation(1:floor(length(permutation)/5)));
	% concat_val = all_processed_clips(permutation(floor(length(permutation)/5+1:end)));



	% Normalise both training and test data
	stest = size(concatTestArray);
	strain = size(concatTrainArray);
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %apply hill function per-frequency:

    concatTestArray = reshape(concatTestArray, nf, nh, stest(2));
    concatTrainArray = reshape(concatTrainArray, nf, nh, strain(2));

    for currf = 1:nf
        this_freq_train = squeeze(concatTrainArray(currf,:,:));
        this_freq_test = squeeze(concatTestArray(currf,:,:));

        
        if opts.applyHill
    	    if opts.verbose
		    	display('Applying hill function');
		    	tic;
		    end
		    %Apply the hill function separatley to each frequency:
		    %Apply to training data:
	        this_freq_median = median(this_freq_train(:));
	        this_freq_train = 0.02*this_freq_train./this_freq_median;
	        this_freq_train = this_freq_train./(this_freq_train + 1);
	        % this_freq_train = array_fun(@(x) (x./(x+1)),this_freq_train);

		    %Apply to test data
	        this_freq_test = 0.02*this_freq_test./this_freq_median;
	        this_freq_test = this_freq_test./(this_freq_test + 1);
	        % this_freq_test = array_fun(@(x) (x./(x+1)),this_freq_test);
	    end

        concatTrainArray(currf,:,:) = this_freq_train;
        concatTestArray(currf,:,:) = this_freq_test;
        
        if opts.normtype == 1
        	mutemp = mean(this_freq_train(:)); 
        	stdtemp = std(this_freq_train(:)); 
        	concatTrainArray(currf, :,:) = (concatTrainArray(currf, :,:) - mutemp)./stdtemp;
        	concatTestArray(currf, :,:) = (concatTestArray(currf,:,:) - mutemp)./stdtemp;
        end
    end
    
    concattrain = reshape(concatTrainArray, strain(1), strain(2));
    concattest = reshape(concatTestArray, stest(1), stest(2));
    if opts.verbose
    	toc;
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


	if opts.normtype == 0
		mu_train = mean(concattrain(:)); 
		std_train = std(concattrain(:));
		concattrain = (concattrain - mu_train)./std_train;
		concattest = (concattest - mu_train)./std_train;
	end

    
	%     %% %%%% For debugging
	%     concattrain = reshape(concattrain, numfreq, n_h, strain(2));
	% 
	%     figure(2)
	%     for ii = 1:strain(2)
	%         subplot(1,3,1), imagesc(squeeze(prefilt_trainarray(:,:,ii)));
	%         subplot(1,3,2), imagesc(squeeze(concatTrainArray(:,:,ii)));
	%         subplot(1,3,3), imagesc(squeeze(concattrain(:,:,ii)));
	%         drawnow;
	%         pause(0.5)
	%     end
	%     %%
	% concattrain = reshape(concatTrainArray, strain(1), strain(2));
	% concattest = reshape(concatTestArray, stest(1), stest(2));

catch ME
    display(getReport(ME));
end
