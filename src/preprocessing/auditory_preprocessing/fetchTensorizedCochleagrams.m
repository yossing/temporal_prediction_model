function [tensorized_cochleagrams, settings] = fetchTensorizedCochleagrams(cochleagram_save_dir, sound_file_dir, varargin)
%Function to create tensorized cochleagrams from folder of sound clips or to load
%the cochleagrams if they already exist
%Inputs:
%-cochleagram_save_dir: Path to folder where concatenated tensorized 
%                       cochleagrams of all files should be saved. If there is 
%                       already a saved cochleagram array in this location and
%                       'create_cochleagrams' is set to false, the saved cochleagram
%                       will be used for remainder of preprocessing
%-sound_file_dir   : This field can be left empty if the cochleagrams are being loaded

%-varargin. Possible arguments include:
%---overwrite_cochleagrams. If true, will overwrite existing cochleagram if exists
%                        If false, cochleagram will only be created if a saved 
%                        version does not exist in cochleagram_save_dir 
%---maxFilesToProcess = 20000; %Limit the number of files to wieldxy amount of data


%*********************Process input args and set defaults***********************
opts.overwrite_cochleagrams = false; %If true, will overwrite existing cochleagram if exists
% opts.cochleagram_varargin = [];
opts.maxFilesToProcess = 20000;
% opts.cochleagram_save_name = 'all_concattens.mat';

%Parse through varargin and replace default parameters where necessary
[opts, varargin] = vl_argparse(opts, varargin) ;
%********************Create or load tensorized cochleagrams*********************
if (~exist([cochleagram_save_dir,'all_concattens.mat'],'file'))||opts.overwrite_cochleagrams
    display('Creating array of tensorized cochleagrams');
    tic;
    [mf,settings] = makeTensorizedCochleagrams(sound_file_dir,...
                                               cochleagram_save_dir,...
                                               varargin);

    display('Completed creating tensorized cochleagrams');
    toc;
else %load existing cochleagrams
    display('Loading matfile containing tensorized cochleagrams')
    tic;
    mf = matfile([cochleagram_save_dir,'all_concattens.mat']);
    display('Completed loading matfile with tensorized cochleagrams');
    toc;
    settings = load([cochleagram_save_dir,'settings.mat']);
end

fss = mf.Fss;
nFiles = length(fss);
nFiles = min(nFiles,opts.maxFilesToProcess);
display('Loading all_concattens array from matfile');
tic;
tensorized_cochleagrams = mf.all_concattens(1,1:nFiles);
display('completed loading all_concattens from matfile');
toc;
