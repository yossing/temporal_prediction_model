clear;
clc;

%*******************************Set datapaths***********************************
ds_name = 'all_sounds';
sound_file_dir = '/path/to/folder/with/.wav/files/';
cochleagram_save_dir = '/path/to/intermediary/save/folder/';
final_save_dir = '/path/to/final/save/folder/';
if ~exist(cochleagram_save_dir,'dir')
    mkdir(cochleagram_save_dir);
end
if ~exist(final_save_dir,'dir')
    mkdir(final_save_dir);
end

%*******************************Make and save stimuli***************************
success = makeAndSaveSoundStimuli(cochleagram_save_dir, final_save_dir, ds_name,...
                                  sound_file_dir)