%Author: Yosef Singer
function g = evalGaussian(params,width,height)

    [xi,yi] = meshgrid(1:width,1:height);
    xi = xi(:);
    yi=yi(:);

    A = params(1);
    x0 = params(2);
    y0 = params(3);
    theta = params(4);
    sigma_x = params(5);
    sigma_y = params(6);
    
    xip =  (xi-x0)*cos(theta) + (yi-y0)*sin(theta);
    yip = -(xi-x0)*sin(theta) + (yi-y0)*cos(theta);
    
    g = A*exp(-0.5*(((xip.^2)/(sigma_x^2))+((yip.^2)/(sigma_y^2))));
    g = reshape(g,width,height);

end
