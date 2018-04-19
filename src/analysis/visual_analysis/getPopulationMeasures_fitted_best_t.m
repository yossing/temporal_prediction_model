%Author: Yosef Singer

function [sfs,thets,x0,y0,nx,ny,r2mask, r2, fitted_Gs, sigma_x, sigma_y] = getPopulationMeasures_fitted_best_t(gabor_params,r2,fitted_Gs)

if size(r2,1)==1
    r2=r2';
end
sfs = squeeze(gabor_params(:,7));%Normalise to deg per pixel
sfs = abs(sfs);
thets = squeeze(gabor_params(:,4));
%put in range 0-180 deg
thets = wrapTo2Pi(thets);
thets(thets>pi)=thets(thets>pi)-pi;
sigma_x = squeeze(gabor_params(:,5));
sigma_y = squeeze(gabor_params(:,6));
phi = squeeze(gabor_params(:,8));
x0 = squeeze(gabor_params(:,2));
y0 = squeeze(gabor_params(:,3));

nx = abs(sigma_x).*abs(sfs);
ny = abs(sigma_y).*abs(sfs);

sigma_x_mask = abs(sigma_x)>0.5;
sigma_y_mask = abs(sigma_y)>0.5;
large_sigma_x_mask = abs(sigma_x)<10;
large_sigma_y_mask = abs(sigma_y)<10;
x0_mask= x0>0 & x0<20;
y0_mask= y0>0 & y0<20;


display('number of units excluded by sigma<1 mask: ')
length(sigma_x_mask) - sum((sigma_x_mask)&(sigma_y_mask))

nx_mask = abs(nx)<1.5;
ny_mask = abs(ny)<1.5;


r2mask = r2>0.7;
disp('number of units excluded by r2 mask: ')
length(r2mask) - sum(r2mask)
r2mask = r2mask & abs(sfs)<0.5;
display('number of units excluded by abs_sfs<0.5 mask: ')
length(abs(sfs)<0.5) - sum(abs(sfs)<0.5)
display('number of units excluded by x0 (spatial position) mask: ')
length(x0_mask) - sum(x0_mask)

% sum(abs(sfs(:))>0.5)
display('Selection criterion: r^2>0.7 & sfs<0.5');
display('sigma_x or sigma_y < 1')
display('x0<0 or x0>20')

r2mask = r2mask & (sigma_x_mask & sigma_y_mask) &x0_mask &y0_mask;
disp('total units excluded: ')
length(r2mask) - sum(r2mask)

disp('total units:' )
length(r2mask)

