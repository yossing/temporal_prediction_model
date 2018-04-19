function [mf,settings] = makeTensorizedCochleagrams(file_dir,save_dir,varargin)
%Function to processes all .wav files in file_dir and converts them to cochleagrams 
%wThe cochleagrams are then tensorized with n_h history bins. 
%The tensorized cochleagrams of all files are concatenated into a single cell array
%which is saved incrementally to a matfile as each batch of sound files is processed.
%The matfile is named all_concattens (this is changeable, but note that other functions
%rely on this variable name being consistent). 
%The all_concattens matfile is created in the specified save_dir
%The settings are also saved separately to the same directory 

%Possible varargin: verbose, settings, cochleagram_type, Fs, maxFilesToProcess,applyHill
%---cochleagram_type: %0->normal, 1->power
%*********************Process input args and set defaults***********************
opts.verbose = true;

settings.n_h = 43;
settings.numfreq = 32;
settings.numpredict = 3;
settings.numsecs = 2;
settings.maxfreq = 500*(2^(5+(1/6)));
settings.minfreq = 500;
settings.dt=5; %ms
settings.spacing_type = 'log';

%Could also pass in entire 'settings' struct as a varargin
opts.settings = settings;
clear settings;

opts.save_name = 'all_concattens.mat';
opts.verbose = true;
opts.maxFilesToProcess = 20000;
opts.Fs = 44100;
opts.cochleagram_type = 1; %0 for normal, 1 for power
%% if cochleagram_type == 1
%%     opts.applyHill = true;
%% else
%%     opts.applyHill = false;
%% end
%%Parse through varargin and replace default parameters where necessary
if length(varargin)==1&&isempty(varargin{1}),varargin = {};end;
[opts, varargin] = vl_argparse(opts, varargin) ;
%*******************************************************************************

%Save settings for cochleagram in save directory
settings = opts.settings;
save([save_dir,'settings.mat'],'settings');

rng('shuffle')


d = dir([file_dir '*.wav']);

numfiles = numel(d);
if numfiles ==0
    error('No .wav files exist in specified directory');
end
numFilesToProcess = min(opts.maxFilesToProcess, numfiles);


%*******************************************************************************
%Now loop through nFiles sound files in the directory and create cochleagrams
tic;
if opts.verbose;display('Starting sound preprocessing...');end;
try
    mf = matfile([save_dir,opts.save_name],'Writable',true);
    mf.all_concattens = {};
    mf.Fss= [];
    all_concattens_ix = 1;
    %break into batch and process and save each batch seperately
    if numFilesToProcess>500
    num_batches = 500;
    else
        num_batches = 1;
    end
    batch_size = floor(numFilesToProcess/num_batches);
    for batch = 1:num_batches
        start_file_idx = 1+(batch-1)*batch_size;
        end_file_idx = start_file_idx+batch_size-1;
        these_file_idx = start_file_idx:1:end_file_idx;
        
    % parfor_progress(length(these_file_idx));
    for ii = 1:length(these_file_idx)
        file_idx = these_file_idx(ii);

        name = d(file_idx).name;
        assert(~isempty(strfind(name, '.wav'))) %Shouldn't fail because we only included files in dir with .wav extension.

            [sound, this_Fs] = audioread([file_dir, name]);
    %         if verbose
    %             display(['Processing ',name,'. Original sampling rate: ',num2str(this_Fs)]);
    %         end
            if this_Fs >= opts.Fs
                %if the sampling rate is higher than this, then resample
                if this_Fs>opts.Fs; sound = resample(sound, opts.Fs,this_Fs);end;
                Fss(ii) = this_Fs; 
                this_Fs = floor(this_Fs);
                if size(sound,2)>1; sound = sound(:,1);end %stereo->mono

                %First convert the entire soundclip into a cochleagram:
                if opts.cochleagram_type == 0 %normal
                [X_ft, t, params] = cochleagram(sound, this_Fs, settings.dt, settings.spacing_type, settings.minfreq, settings.maxfreq, settings.numfreq); 
                elseif opts.cochleagram_type == 1 %power
                [X_ft, t, params] = power_cochleagram(sound, this_Fs, settings.dt, settings.spacing_type, settings.minfreq, settings.maxfreq, settings.numfreq); 
                end
                tensor = tensorize(X_ft, settings.n_h); 
                if size(tensor,3)>settings.n_h
                    tensornopad = tensor(:,:,settings.n_h:end);
                    %eliminate clips where all frequencies have no variance
                    freq_var = var(tensornopad,0,1);%look along first dimension.
                    zero_var_mask = squeeze(freq_var)==0;
                    keep_ts = ~any(zero_var_mask,1);
                    valid_tensor = tensornopad(:,:,keep_ts);

                    % %apply hill function to power spectrogram
                    % if opts.applyHill
                    %     valid_tensor = arrayfun(@(x) (x./(1+x)),valid_tensor);
                    % end
                    concattens{ii} = valid_tensor;
                end 
            end %if has correct sampling rate
        end %if file is .wav
    %     parfor_progress;
    end %parfor parse over files
    % parfor_progress(0);
    valid_idx = find(~cellfun(@isempty,concattens));
    concattens = concattens(valid_idx);
    Fss= Fss(valid_idx);

    curr_size = length(valid_idx);
    mf.all_concattens(1,all_concattens_ix:all_concattens_ix+curr_size-1) = concattens;
    mf.Fss(1,all_concattens_ix:all_concattens_ix+curr_size-1) = Fss;
    clear Fss concattens;

    all_concattens_ix = all_concattens_ix+curr_size;

    if verbose
        display(['Completed ',num2str(100*all_concattens_ix/numfiles),'% of preprocessing']);
        display(['Sound files processed:',num2str(all_concattens_ix)]);
    %         fprintf('Size of soundCellArray: ');ByteSize(soundCellArray);
        toc;
        fprintf('\n');
    end %if verbose

    %save a variable indicating the number of files processed in the matfile
    mf.nFiles = all_concattens_ix;
% end %try
catch ME
    display(getReport(ME));
end
