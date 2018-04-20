%Author: Yosef Singer

function [f, df] = spatialGaborCost(params,v,regul_params)

if nargin>2
    for ii =1:length(regul_params)
        eval(['lam',num2str(ii),'=regul_params(ii);']);
    end
end
        

[xi,yi] = meshgrid(1:size(v,1),1:size(v,2));
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

xip =  (xi-x0)*cos(theta) + (yi-y0)*sin(theta);
yip = -(xi-x0)*sin(theta) + (yi-y0)*cos(theta);
p = exp(-((xip.^2)/2/sigma_x^2)-((yip.^2)/2/sigma_y^2));

q = cos(2*pi*xip*freq+phase);
r = sin(2*pi*xip*freq+phase);
v_hat = A*p.*q;

v_diff = v_hat-v(:);    

%Use the square error to calculate the cost
f = sum(sum((v_diff).^2));
%Add regularisation factor to minimise the sf and sigma values
f = f + lam1*sum(freq(:).^2)+lam2*sum(sigma_x(:).^2)+lam2*sum(sigma_y(:).^2);
%Calculate the derivatives
df_dA = 2*v_diff.*v_hat/A;
df_dx0 = 2*v_diff*A.*p.*(q.*((xip./(sigma_x.^2)).*cos(theta)-(yip./(sigma_y.^2)).*sin(theta))+2.*pi.*freq.*r.*cos(theta));
df_dy0 = 2*v_diff*A.*p.*(q.*((xip./(sigma_x.^2)).*sin(theta)+(yip./(sigma_y.^2)).*cos(theta))+2.*pi.*freq.*r.*sin(theta));
% df_dtheta = 2*v_diff.*A.*p.*yip.*(q.*xip.*((1/(sigma_x.^2))+(1/(sigma_y.^2)))-2*pi*freq.*r);

df_dtheta = 2*v_diff.*A.*p.*yip.*(q.*xip.*((1/sigma_y.^2)-(1/sigma_x.^2))-2*pi*freq*r);

% df_dsigmax = sign(sx).*2.*v_diff.*v_hat.*(xip.^2)./(sigma_x.^3);
% df_dsigmay = sign(sy).*2.*v_diff.*v_hat.*(yip.^2)./(sigma_y.^3);
df_dsigmax = 2.*v_diff.*v_hat.*(xip.^2)./(sigma_x.^3);
df_dsigmay = 2.*v_diff.*v_hat.*(yip.^2)./(sigma_y.^3);

% df_dfreq = sign(sf).*-2.*v_diff*A.*p.*r*2*pi.*xip;
df_dfreq = -2.*v_diff*A.*p.*r*2*pi.*xip;
df_dphase = -2*v_diff*A.*p.*r;

% add derivatives of regularisation factors:
% df_dfreq = sum(df_dfreq(:)) +sign(sf).*sum(2.*lam1.*freq(:));
df_dfreq = sum(df_dfreq(:)) +sum(2.*lam1.*freq(:));
% df_dsigmax = sum(df_dsigmax(:)) +sign(sx).*sum(2.*lam2.*sigma_x(:));
% df_dsigmay = sum(df_dsigmay(:)) +sign(sy).*sum(2.*lam2.*sigma_y(:));
df_dsigmax = sum(df_dsigmax(:)) +sum(2.*lam2.*sigma_x(:));
df_dsigmay = sum(df_dsigmay(:)) +sum(2.*lam2.*sigma_y(:));
df = [sum(df_dA(:)),sum(df_dx0(:)),sum(df_dy0(:)),sum(df_dtheta(:)),...
      sum(df_dsigmax(:)),sum(df_dsigmay(:)),sum(df_dfreq(:)),...
      sum(df_dphase(:))]';
% end
end


