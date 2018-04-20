%Author: Yosef Singer
%Create an array of randonly generated Gabor functions
clear;
RF_size = 20;
numRF = 30;
%Generate some random gabors
for ii = 1:numRF
A(ii) = 0.5 + 5*rand;
x0(ii) = 0.5*RF_size+abs(0.1*RF_size*randn);
y0(ii) = 0.5*RF_size+abs(0.1*RF_size*randn);
theta(ii) = 2*pi*rand;
sigma_x(ii) = 0.5+abs(5*rand);
sigma_y(ii) = 0.5+abs(5*rand);
sf(ii) = abs(0.01+0.3*rand);
P(ii) = 2*pi*rand;
gabor_params(ii,:) = [A(ii),x0(ii),y0(ii),theta(ii),sigma_x(ii),sigma_y(ii),sf(ii),P(ii)]';
gabor = evalSpatialGabor(gabor_params(ii,:),RF_size,RF_size);
gabors(ii,:,:)= reshape(gabor,RF_size,RF_size);
end
tic;
fit_options.num_sg_runs = 12;
fit_options.num_g_runs = 20;
fit_options.display = 2;
fit_options.regul_params = [10e-3,10e-6];

% minimizer = @(cost_fn,init_params,options)minFunc(cost_fn,init_params,options);
minimizer = @(cost_fn,init_params,options)fminunc(cost_fn, init_params,options);
% parfor_progress(numRF);
for ii = 1:numRF  
    display(['True sx,sy: ',num2str(sigma_x(ii)),',',num2str(sigma_y(ii))]);
fit_results(ii) = doGaborFits(squeeze(gabors(ii,:,:)),minimizer,fit_options);
% parfor_progress;
end
% parfor_progress(0);
toc
%%
for ii =1:numRF
    fitted_params(ii,:) = fit_results(ii).gabor_params;
    f_val(ii) = fit_results(ii).sse;
    fitted_gabors(ii,:,:) = fit_results(ii).fitted_gabor;
    p0(ii,:) = fit_results(ii).p0;
    r2(ii,:) = fit_results(ii).r2;
    var_unxpl(ii,:) = fit_results(ii).var_unxpl;
end


%% Compare fits and original Gabors
mask = ones(numRF,1);
% mask = f_val>0.005;
sum(mask)
dispgrid_size = ceil(sqrt(numRF));
gabor_image = zeros(dispgrid_size*RF_size,dispgrid_size*RF_size);
fit_image = zeros(dispgrid_size*RF_size,dispgrid_size*RF_size);

for ii = 1:numRF
    x_offset = RF_size*mod(ii-1,dispgrid_size)+1;
    y_offset = RF_size*floor((ii-1)/dispgrid_size)+1;
    this_gabor = squeeze(gabors(ii,:,:));
%     this_gabor = this_gabor/max(abs(this_gabor(:)));
    gabor_image(y_offset:y_offset+(RF_size-1), x_offset:x_offset+(RF_size-1)) = ...
    mask(ii)*this_gabor; 
    this_fit = squeeze(fitted_gabors(ii,:,:));
    fit_image(y_offset:y_offset+(RF_size-1), x_offset:x_offset+(RF_size-1),:) = ...
    mask(ii)*this_fit;
end

%% compare fits to original weights
figure(12);
mx = max(abs(gabor_image(:)));
subplot(1,3,1),imagesc(gabor_image,[-mx,mx]);
subplot(1,3,2),imagesc(fit_image,[-mx,mx]);
subplot(1,3,3),imagesc(gabor_image-fit_image)%,[-mx,mx]);
%View the distribution of spatial frequencies
%%
figure(13);
subplot(1,4,1),histogram(sf)
subplot(1,4,2),histogram(p0(:,7))
subplot(1,4,3),histogram(abs(fitted_params(:,7)));%,'BinLimits',[0,1])
% subplot(1,4,4),histogram(((abs(gabor_params(:,7))-abs(p0(:,7)))./gabor_params(:,7)),'BinLimits',[-0.5,0.5])
subplot(1,4,4),histogram(((abs(gabor_params(:,7))-abs(fitted_params(:,7)))./gabor_params(:,7)))%,'BinLimits',[-0.5,0.5])

% subplot(1,3,3),histogram(abs(fitted_params(:,7)),'BinLimits',[0,0.5])
%%
figure(13);
subplot(1,4,1),histogram(sigma_x,5)
subplot(1,4,2),histogram(p0(:,5),5)
subplot(1,4,3),histogram(abs(fitted_params(:,5)),5)%,'BinLimits',[0,0.5])
subplot(1,4,4),histogram(((abs(gabor_params(:,5))-abs(fitted_params(:,5)))./gabor_params(:,5)))%,'BinLimits',[-0.5,0.5])

%%
figure(14)
subplot(1,3,1),histogram(A)
subplot(1,3,2),histogram(A'-fitted_params(:,1),'BinLimits',[-5,5])
subplot(1,3,3),histogram(fitted_params(:,1),'BinLimits',[-5,5])
%%
figure(14)
subplot(1,3,1),histogram(wrapTo2Pi(P))
subplot(1,3,2),histogram(wrapTo2Pi(fitted_params(:,8)));%,'BinLimits',[-5,5])
subplot(1,3,3),histogram(abs(rad2deg(wrapTo2Pi(fitted_params(:,8))-wrapTo2Pi(P)')),0:20:360);

%%
plt_count = sum(fitted_params(:,7)>0.5);
gridsize = ceil(sqrt(plt_count));
selected_idx = find(fitted_params(:,7)>0.5);
% figure(15);
for ii = 1:plt_count
        subplottight(gridsize,gridsize,ii);
%         imagesc(squeeze(gabors(selected_idx(ii),:,:)));
        imagesc(squeeze(fitted_gabors(selected_idx(ii),:,:)));
end
%% compare A, SF, theta and phase of data and fits
mask = find(gabor_params(:,7)<1);
[gabor_params(mask,1).*1e-3,fitted_params(mask,1).*1e-3,...
gabor_params(mask,7),fitted_params(mask,7),...
rad2deg(wrapTo2Pi(gabor_params(mask,4))),rad2deg(wrapTo2Pi(fitted_params(mask,4))),...
rad2deg(wrapTo2Pi(gabor_params(mask,8))),rad2deg(wrapTo2Pi(fitted_params(mask,8)))]
scatter(rad2deg(wrapTo2Pi(gabor_params(:,4))),rad2deg(wrapTo2Pi(fitted_params(:,4))))
