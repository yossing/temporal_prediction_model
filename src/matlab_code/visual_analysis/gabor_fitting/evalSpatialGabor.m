%Author: Yosef Singer
function g = evalSpatialGabor(params,width,height)

[xi,yi] = meshgrid(1:width,1:height);
xi = xi(:);
yi=yi(:);

A = params(1);
x0 = params(2);
y0 = params(3);
theta = params(4);
sigma_x = params(5);
sigma_y = params(6);
lambda = 1/params(7);
phase = params(8);
xt = (xi-x0);
yt = (yi-y0);

xip =  xt*cos(theta) + yt*sin(theta);
yip = -xt*sin(theta) + yt*cos(theta);

g = A*exp(-((xip.^2)/2/sigma_x^2)-((yip.^2)/2/sigma_y^2)).*cos(2*pi*xip/lambda+phase);
g = reshape(g,width,height);
end
