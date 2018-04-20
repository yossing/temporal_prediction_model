%Author: Yosef Singer

function[new_gabor_params, new_fitted_Gs, new_r2] = fixAliasedSFs_fitted_best_t(gabor_params, best_weights, fitted_Gs,r2)

old_sf = gabor_params(:,7);
phi = gabor_params(:,8);

%Let's keep track of the units that are modified
mod_mask = zeros(size(old_sf));

%first make any negative sfs positive:
neg_mask = old_sf<0;
old_sf(neg_mask) = -old_sf(neg_mask);
phi(neg_mask) = -phi(neg_mask);

% mod_mask = mod_mask|neg_mask;
clear neg_mask;
%Now look for sfs>0.5 -> Aliased
aliased_mask = old_sf>0.5;
mod_mask = mod_mask|aliased_mask;

new_sf = old_sf;
new_sf(aliased_mask) = 1-old_sf(aliased_mask);

%Again, look for any negative sfs
% neg_mask = new_sf<0;
% new_sf(neg_mask) = - new_sf(neg_mask);
% phi(neg_mask) = -phi(neg_mask);

% mod_mask = mod_mask|neg_mask;
clear neg_mask;

new_gabor_params = gabor_params;
new_gabor_params(:,7) = new_sf;
new_gabor_params(:,8) = phi;


height = size(best_weights,2);
width = size(best_weights,3);

new_fitted_Gs = fitted_Gs;
sum(mod_mask)
for ii = 1:size(gabor_params,1)

        new_fitted_Gs(ii,:,:) = evalModifiedSpatialGabor(new_gabor_params(ii,:),...
                                                            width, height, mod_mask(ii));
%             end
        this_weight= best_weights(ii,:,:);
        temp_r2 = corrcoef(squeeze(new_fitted_Gs(ii,:,:)),this_weight);
        this_weight= best_weights(ii,:,:);
        temp_r2 = corrcoef(new_fitted_Gs(ii,:,:),this_weight);
        new_r2(ii) = temp_r2(2);
        if isnan(new_r2(ii))
            new_r2(ii) = 0;
        end
        if new_r2(ii)<0.2
            new_r2(ii)-r2(ii)
        end
%             if new_r2(ii,jj)<0
%                subplot(1,3,1); imagesc(squeeze(new_fitted_Gs(ii,:,:)));
%                subplot(1,3,2); imagesc(squeeze(best_weights(ii,:,:)));
%                subplot(1,3,3); imagesc(squeeze(new_fitted_Gs(ii,:,:)-best_weights(ii,:,:)));
%                drawnow;
%                pause(1);
%             end
end
new_r2 = squeeze(new_r2);
sum(new_r2<0.2)
sum(r2<0.2)


    function g = evalModifiedSpatialGabor(params,width,height,modified)

        [xi,yi] = meshgrid(1:width,1:height);
        xi = xi(:);
        yi=yi(:);

        A = params(1);
        x0 = params(2);
        y0 = params(3);
        theta = params(4);
        sigma_x = params(5);
        sigma_y = params(6);
        freq = params(7);
        phase = params(8);
        xt = (xi-x0);
        yt = (yi-y0);

        xip =  xt*cos(theta) + yt*sin(theta);
        yip = -xt*sin(theta) + yt*cos(theta);
        
        if modified
%             freq = 1-freq;
            phase_mod = -2*pi*xip - phase;
        else
            phase_mod = phase;
        end
        g = A*exp(-((xip.^2)/2/sigma_x^2)-((yip.^2)/2/sigma_y^2)).*cos((2*pi*xip*freq)+phase_mod);
        g = reshape(g,width,height);
    end
end
