function success = makeAndSaveSoundStimuli(cochleagram_save_dir, final_save_dir, ds, sound_file_dir, varargin)
%********************************Set options***********************************
%options
opts.verbose = true;
opts.create_subrefs = true;
opts.maxFilesToProcess = 20000;
opts.overwrite_cochleagrams = false;
% opts.cochleagram_varargin = [];
[opts, varargin] = vl_argparse(opts, varargin) ;

%***************************Fetch cochleagrams**********************************
[tensorized_cochleagrams, settings] = fetchTensorizedCochleagrams(cochleagram_save_dir,...
                                        sound_file_dir);

%***************************Finalise dataset************************************
tic;
if opts.verbose
    display(['starting finalisation of dataset: ',ds]);
end
[concattrain,concattest] = finaliseAuditoryDataset(tensorized_cochleagrams,...
                                                   varargin);

%*******************************Save dataset************************************
final_save_path = [final_save_dir,'training_test_data_',ds,'.mat'];

%% Now make subrefs:
if opts.create_subrefs 
    subrefs = createSoundSubrefs(concattrain);
    clear concattrain;
    %% Finaly save data
    save(final_save_path,'subrefs','concattest','-v7.3');
    clear subrefs concattest;
else
    %% Finaly save data
    save(final_save_path,'concattrain','concattest','-v7.3');
end
if opts.verbose
    display('completed finalisation of dataset');
    toc;
end

save([final_save_dir,'settings.mat'],'settings');

success = true;