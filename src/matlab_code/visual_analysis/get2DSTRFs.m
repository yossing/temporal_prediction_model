%Author: Yosef Singer

function vis_strf = get2DSTRFs(X0,Y0,Theta,weights)

numweights = size(weights,1);
num_tsteps = size(weights, 4);
RF_size = size(weights, 2);
try
    
parfor ii = 1:numweights
    for jj = 1:num_tsteps
    %Center and rotate according to parameters of best timestep
    x0 = X0(ii);
    y0 = Y0(ii);
    theta= Theta(ii);
    this_weight = squeeze(weights(ii,:,:,jj));
    %first translate the image matrix by x0,y0
    A = [1 0 0; 0 1 0; x0 y0 1];
    t_tform = affine2d(A);
    t_weight = imwarp(this_weight,t_tform);
    %Then rotate the matrix by theta
    theta =  abs(theta);
    B = [cos(theta) sin(theta) 0; -sin(theta) cos(theta) 0;0 0 1];
    r_tform = affine2d(B);    
    r_warp = imwarp(t_weight,r_tform);

    if size(r_warp,1)>RF_size
        this_strf = trapz(r_warp(2:21,2:21),1);
    else
        this_strf = trapz(r_warp,1);
    end        
    vis_strf(ii,:,jj) = this_strf;
    end
end
catch ME
    getReport(ME);
end


