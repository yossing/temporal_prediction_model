%Author: Yosef Singer

function output = doGaborFits(input_data,minimizer,options)
%A function to fit a 2D Gabor. Avoids local minima by first fitting a
%spectral Gabor (effectively 2 gaussians) to the 2D FT of the input. Also
%use regularisation parameters in the spatial cost function ot keep the
%the spatial frequency and sigma values small. 

%Inputs
%input_data: an mxn array of data to which gabor is being fit

%minimizer: function handle to Miminzation function. This can be any 
%unconstrained minimizer such as fminunc or minFunc

%options: a struct with the following possible fields:
%-display: an integer with the following possible values:
%---0 -> display off
%---1 -> show summary and summary plots at the end of fit
%---2 -> show same as 1 as well as the fit in the spectral domain
%-num_sg_runs: an integer value. The number of times the spectral gabor is
%              fit using estimated initial values and added noise. At the
%              end of all the runs the fit with the smallest cost is taken.
%-num_g_runs: an integer value. As with num_sg_runs, but for spatial gabor.
%-regul_params: Regularisation parameters. One for spatial frequency, and one for sigmax and sigmay.
%               The reason for these regularistaion factors is that we want to keep the 
%               values of these parameters as small as possible (within limits). You may
%               want to play around with these values to ensure your fits are correctly
%               parameterised. 
%               Setting these values to 0 (or very small) will mean they
%               have no effect on the fit.
%-minimizer_options: a struct with any of the valid fields of options passed
%                  into minimizer for both spectral and spatial Gabor fits.


%% Process inputs
if nargin<2 || isempty(minimizer)
    %If no minimzation function is passed in, use MATLAB's fminunc
    minimizer = @(cost_fn,init_params,options)fminunc(cost_fn,...
                                                      init_params,options);
end
if nargin<3 || isempty(options)
    options = struct;
end
    
if isfield(options,'display')
    disp = options.display;
else
    disp = 0;
end
if isfield(options,'num_sg_runs')
    num_sg_runs = options.num_sg_runs;
else
    num_sg_runs = 10;
end
if isfield(options,'num_g_runs')
    num_g_runs = options.num_g_runs;
else
    num_g_runs = 100;
end
if isfield(options,'regul_params')
    regul_params = options.regul_params;
else
    regul_params = [10e-3,10e-6];
end
if isfield(options,'minimizer_options')
    minimizer_options = options.minimizer_options;
else
    minimizer_options.Display = 'off';
end

%% Fit the gabor in the spectral domain 
% This gives us a good idea of the starting  parameters for fitting in the 
% spatial domain.

%Start by taking the Fourier transform
H_s = fftshift(fft2(input_data));
%Take only the real part
abs_H_s = abs(H_s); 
%Interpolate to get better resolution
abs_H_s = interp2(abs_H_s,4);
interp_width = size(abs_H_s,1);
interp_height = size(abs_H_s,2);
%Find the peak of the Gaussian in Fourier domain
[r,c,maxval] = max2(abs_H_s);
u = length(abs_H_s)/2+1-c;
v = length(abs_H_s)/2+1-r; 
%Calculate the halfwidth of the spatial frequency (by summing the elements
%larger than 0.5*maxval) to get an indication of the width of each Gaussian
sf_hw = sqrt(sum(sum(abs_H_s>(0.5*maxval)))/(2*pi));

% Now fit the spectral Gabor - this is effectively the sum of two Gaussians
a = 1/sf_hw;
b = a;
sg_thet = 0; 
K = maxval;
sg_init = [K,u,v,sg_thet,a,b]';
%Get an idea of what the spectral Gabor looks like prior to fitting
sg0 = evalSpectralGabor(sg_init,interp_width,interp_height);
sg_f0 = sum((abs_H_s(:)-sg0(:)).^2); 

%This allow us to deal with fixed variabls such as abs_H_s without passing
%them around during the fitting.
spectralCostFunc =  @(params)spectralGaborCost(params,abs_H_s);


sg_results = zeros(num_sg_runs,length(sg_init));
%Can easily make parallel by using parfor
parfor ii = 1:num_sg_runs
% for ii = 1:num_sg_runs
K0 = K+ abs(rand);
% u0 = u+0.1*length(abs_H_s)*randn;
% v0 = v+0.1*length(abs_H_s)*randn;
u0=u;
v0=v;
sg_thet0 =sg_thet + rand*pi; %Randomly pick from uniform distribution;
% a0 = abs(a+1/0.1*length(abs_H_s)*randn);
% b0 = abs(b+1/0.1*length(abs_H_s)*randn);
a0=a;
b0=b;
sg_init = [K0,u0,v0,sg_thet0,a0,b0]';

[sg_results(ii,:),sg_f(ii),exitflag,output] = minimizer(spectralCostFunc,sg_init,minimizer_options);
end

%% Process the outputs from the spectral fit
[sgf_val,sg_best_run] = min(sg_f);
sg_params = sg_results(sg_best_run,:);
sgg = evalSpectralGabor(sg_params,interp_width,interp_height);

if disp ==2
    figure(101);
    subplot(1,3,1),imagesc(abs_H_s);
    title('2D Fourier transform of input');
    subplot(1,3,2),imagesc(sg0);
    title('Initial estimate of Spectral Gabor');
    subplot(1,3,3),imagesc(sgg);
    title('Best fitted Spectral Gabor');
    drawnow;
end
u = sg_params(2);
v = sg_params(3);
a = sg_params(5);
b = sg_params(6);
sigma_x = 1/sqrt(pi*abs(a));
sigma_y = 1/sqrt(pi*abs(b));
% sigma_x = 1/abs(a)
% sigma_y = 1/abs(b)

%Because we get the parameters from the interploated FFT, we need to
%multiply the parameters that rely on U and V by a scaling factor 
input_width = size(input_data,1);
input_height = size(input_data,2);
x_scale = 1/interp_width;
y_scale = 1/interp_height;
%Scale variables
% sigma_x = sigma_x*x_scale*input_width
% sigma_y = sigma_y*y_scale*input_height
u=x_scale*u;
v=y_scale*v;

%Estimate the spatial frequency
sf = sqrt(u^2 + v^2);
%Normalise spatial frequency to cycles per pixel.
% sf = sf/(input_width*input_height)
%Estimate the orientation of the Gabor
theta = atan(v/u);
%Assume that the Gabor is centered in the middle the spatial field:
x0 = 0.5*input_width;
y0 = 0.5*input_height;
A = 1;
phase =0;

%Use these as starting parameters for fit in spatial domain
p0 = [A,x0,y0,theta,sigma_x,sigma_y,sf,phase]';
%% Fit in the spatial domain

%Get an indication of the initial cost before fitting:
g0 = evalSpatialGabor(p0,input_width,input_height);
f0 = sum((input_data(:)-g0(:)).^2); 

spatialCostFunc = @(params)spatialGaborCost(params,input_data,regul_params);
% minFunc_options.Display = 'off';
% %Let's set some bounds on the values and use minConf instead of minFunc:
% minConf_options.verbose = 0;
% minConf_options.maxIter = 5000;
% % 
% lb = [-100.1,0,0,0,0.1,0.1,0.01,0]';
% ub = [100,40,40,2*pi,1000,1000,0.5,2*pi]';

results = zeros(num_g_runs,length(p0));

%Can easily make parallel by using parfor
parfor ii = 1:num_g_runs
% for ii = 1:num_g_runs
%     init_params = zeros(length(p0),1);
    
    %Add some variance to each of the parameters 
    A_init = abs(A + 2*rand);
    x0_init = x0 + 2*randn;
    y0_init = y0 + 2*randn;
    
    theta_init = theta + (pi/4)*randn;
    sigma_x_init = abs(sigma_x + 2*rand);
    sigma_y_init = abs(sigma_y + 2*rand);
    sf_init = sf + abs(rand);
    %we have no initial estimate of the phase, so draw value from a uniform
    %distribution between -pi and pi
    min_p = 0;
    max_p = 2*pi;
    phase_init = min_p + (max_p-min_p)*rand;
    init_params = [A_init,x0_init,y0_init,theta_init,sigma_x_init,sigma_y_init,sf_init,phase_init]';
    
    [results(ii,:),cost(ii),exitflag(ii),output(ii)] = minimizer(spatialCostFunc,init_params,minimizer_options);

%     [results(ii).params,cost(ii),exitflag(ii),output(ii)] = minFunc(myObj,init_params,minFunc_options);
%     [results(ii).vals,f(ii),funevals(ii)] = minConf_TMP(myObj,init_params,lb,ub,minConf_options);
end
[best_cost,best_run] = min(cost);
% p_out = results(best_run).params;
p_out = squeeze(results(best_run,:));
g = evalSpatialGabor(p_out,input_width,input_height);
fitted_g = g;
%Calculate the fraction of unexplained variance
var_unexplained = var(input_data(:)-g(:))/var(input_data(:));
%Calculate the correlation between the fits and the input_data
temp_r2 = corrcoef(fitted_g(:),input_data(:));
r2 = temp_r2(2);

output = struct;
output.gabor_params = p_out;
output.sse = best_cost;
output.sse0 = f0;
output.fitted_gabor = fitted_g;
output.var_unxpl = var_unexplained;
output.r2 = r2;
output.p0 = p0;
output.param_order = {'A','x0','y0','theta','sigma_x','sigma_y','sf','phase'};


%% Display results
if disp > 0
    mx= max(max(input_data));
    figure(102);
    subplot(1,4,1),imagesc(input_data,[-mx mx]);
    title('Input');
    subplot(1,4,2),imagesc(g0,[-mx mx]);
    title('Estimate of Gabor before fit');
    subplot(1,4,3),imagesc(fitted_g,[-mx mx]);
    title('Fitted Gabor');
    subplot(1,4,4),imagesc(input_data-fitted_g,[-mx mx]);
    title('Residual (Input-fit)');
    display(['r2 value of fit is: ',num2str(r2)]);
    display(['Cost of fit (sse) is: ',num2str(best_cost)]);
    sx_ratio = p_out(5)/sigma_x
    sy_ratio = p_out(6)/sigma_y
    sa_ratio = p_out(5)/a
    sb_ratio = p_out(6)/b
    sx_ratio/sy_ratio
    
    drawnow;
end
function [row, col,maxval] = max2(p)

[maxrows,rowind] = max(p);
% find the best of these maxima
% across the columns
[maxval,col] = max(maxrows);
row = rowind(col);   
end
end