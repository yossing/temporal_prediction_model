function sub_refs = createSoundSubrefs(trainconcat)

D= size(trainconcat,2); %Number of training examples
N = floor(sqrt(D)/10.); % number minibatches

% create the cell array of subfunction specific arguments
sub_refs = cell(N,1);

for i = 1:N
    % extract a single minibatch of training data.
    sub_refs{i} = trainconcat(:,i:N:end);
end   

end