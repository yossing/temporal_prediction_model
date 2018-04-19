%Author: Yosef Singer
function g = evalSpectralGabor(params,width,height)
% [ui,vi] = meshgrid(-0.5*size(v,1)+1:0.5*size(v,1),-0.5*size(v,2)+1:0.5*size(v,2));
[ui,vi] = meshgrid(-0.5*width:0.5*width-1,-0.5*height:0.5*height-1);
% [ui,vi] = meshgrid(1:size(v,1),1:size(v,2));
ui = ui(:);
vi=vi(:);

K = params(1);
u0 = params(2);
v0 = params(3);
theta = params(4);
a = params(5);
b = params(6);
ug =  (ui-u0)*cos(theta) - (vi-v0)*sin(theta);
vg =  (ui-u0)*sin(theta) + (vi-v0)*cos(theta);
%positive guassian:
G_p = exp(-pi*(((ug.^2).*(a^2))+((vg.^2).*(b^2))));
%negative gaussian:
ugg =  (-ui-u0)*cos(theta) - (-vi-v0)*sin(theta);
vgg =  (-ui-u0)*sin(theta) + (-vi-v0)*cos(theta);
G_m = exp(-pi*(((ugg.^2).*(a^2))+((vgg.^2).*(b^2))));
q = (G_p.^2)+(G_m.^2);
g = K*(q.^0.5);
g = reshape(g,width,height);
end
