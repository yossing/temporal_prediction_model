function [concatTestArray, concatTrainArray] = ...
         concatenateSoundArrays(sound,concatTrainArray,...
                                concatTestArray, num_clip_select)                         

[~, num_clips] = size(sound);
if nargin<4 || num_clip_select > num_clips
%    warning(['There are fewer than ',int2str(num_clip_select),'. only adding ',...
%         num2str(num_clips),' to set']);
    num_clip_select = num_clips;
end

%Select 20% for test set
num_test_clips = floor(0.2*num_clip_select);
r = randperm(num_clip_select);
test_ix = r(1:num_test_clips);
train_ix = r(num_test_clips+1:end);

test_set = sound(:,test_ix);
train_set = sound(:,train_ix);

concatTrainArray = cat(2,concatTrainArray, train_set);
concatTestArray = cat(2,concatTestArray, test_set);
