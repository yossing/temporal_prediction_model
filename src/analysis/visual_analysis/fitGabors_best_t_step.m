%Author: Yosef Singer
clear;
clc;
addpath('./gabor_fitting/')

filepath = '/full/path/to/saved/predictive_network.mat'
savepath = '/path/to/save/folder/'
savename = 'gabor_fits.mat';
load(filepath);
weights = network_param_values{1};
weights = double(weights);
RF_size = 20;
clip_length = 7;
sparse_results = false;
if sparse_results
    Wih = Phi';
else
    Wih = theta{1};
end
weights = Wih;
weights = double(weights);

numweights = size(weights,1);
temp_weights = reshape(weights, numweights, RF_size*RF_size*clip_length);
l2_norm = sum(temp_weights.^2,2);
keep_ix = l2_norm(:)>0.1*max(l2_norm(:));
weights = weights(keep_ix,:,:);
% l2_norm = sum(weights.^2,2);
% [sort_vals,sort_ix] = sort(l2_norm,'ascend');
% weights = weights(sort_ix,:);
numweights = size(weights,1);
weights = reshape(weights, numweights, RF_size,RF_size,clip_length);

%% Do gabor fits
% if size(gcp('nocreate'))==0
%     parpool;
% end
% tic;
%%
fit_options.num_sg_runs = 10;
fit_options.num_g_runs = 100;
fit_options.display = 0;
fit_options.regul_params = [10e-3,10e-6];
% fit_options.minimizer_options = ['MaxIter', 100];

% minimizer = @(cost_fn,init_params,options)minFunc(cost_fn,init_params,options);
minimizer = @(cost_fn,init_params,options)fminunc(cost_fn,init_params,options);

%Let's find the best timestep (the one with the highest l2norm) and fit to
%that only:
for ii = 1:numweights
    this_weight =  squeeze(weights(ii,:,:,:));
    this_weight = reshape(this_weight,RF_size*RF_size,7);
    [~,t_best(ii)] = max(sum(this_weight.^2,1));
    best_weights(ii,:,:) = squeeze(weights(ii,:,:,t_best(ii)));
end
numweights = size(best_weights,1);
% parfor_progress(numweights);
display('Fitting Gabors...')

% parfor ii = 1:numweights
for ii = 1:numweights
    this_w = squeeze(best_weights(ii,:,:));    
    vin = this_w;
    fit_results(ii) = doGaborFits(vin,minimizer,fit_options);
    if floor(mod(ii, round(numweights/100)))
        display(['Completed ', int2str((ii/numweights)*100), ' percent of fitting...'])
    end
end

display('Fitting completed! Saving results...')

% toc;
for ii =1:numweights   
    gabor_params(ii,:) = fit_results(ii).gabor_params;
    sse(ii) = fit_results(ii).sse;
    sse0(ii) = fit_results(ii).sse0;
    r2(ii) = fit_results(ii).r2;
    var_unxpl(ii) = fit_results.var_unxpl;
    fitted_Gs(ii,:,:) = fit_results(ii).fitted_gabor;
    p0(ii,:) = fit_results(ii).p0;
end
param_order = fit_results(1).param_order;
theta_filepath = filepath;

%% Save results
save([savepath,savename],'gabor_params','theta_filepath','weights','best_weights','t_best','numweights','RF_size','sse','sse0','fitted_Gs','var_unxpl', 'r2','param_order', 'keep_ix');
