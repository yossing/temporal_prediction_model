%Author: Yosef Singer

function [f, df] = spectralGaborCost(params,v)

% [ui,vi] = meshgrid(-0.5*size(v,1)+1:0.5*size(v,1),-0.5*size(v,2)+1:0.5*size(v,2));
[ui,vi] = meshgrid(-0.5*size(v,1):0.5*size(v,1)-1,-0.5*size(v,2):0.5*size(v,2)-1);
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


%To avoid having the exponential going to 0, which would cause us to divide
%by 0 below (when we calculate dF_ds_temp) 
%positive guassian:
G_p = exp(-pi*(((ug.^2).*(a^2))+((vg.^2).*(b^2))));
%negative gaussian:
um =  (-ui-u0)*cos(theta) - (-vi-v0)*sin(theta);
vm =  (-ui-u0)*sin(theta) + (-vi-v0)*cos(theta);
G_m = exp(-pi*(((um.^2).*(a^2))+((vm.^2).*(b^2)))); %Same formula as G_p

%Evaluate the prediction and the cost:
% q = (G_p.^2)+(G_m.^2)+(2*G_p.*G_m.*cos(2*P));
q = (G_p.^2)+(G_m.^2);%+(2*G_p.*G_m);
G = K*(q.^0.5);

vdiff = G-v(:);
%Use the square error to calculate the cost
f = sum(sum((vdiff).^2));
%Calculate the derivatives
%First the easy ones
dF_dK = 2*vdiff.*(q.^0.5);

dF_dK(isnan(dF_dK)|isinf(dF_dK)) = 0;

dF_dK = sum(dF_dK(:));
% dF_dP = -4*vdiff.*K.*(q.^-0.5).*G_p.*G_m.*sin(2*P);
% dF_dP(isnan(dF_dP)|isinf(dF_dP)) = 0;
% dF_dP = sum(dF_dP(:));
%Now the hard ones
dG_p_da = -2*pi*a.*(ug.^2).*G_p;
dG_m_da = -2*pi*a.*(um.^2).*G_m;
dG_p_db = -2*pi*b.*(vg.^2).*G_p;
dG_m_db = -2*pi*b.*(vm.^2).*G_m;

%These derivatives are the same for (um,vm) and (ug,vg) so use generic form
d_u_du0 = -cos(theta);
d_u_dv0 = sin(theta);
d_v_du0 = -sin(theta);
d_v_dv0 = -cos(theta);

dG_p_du0 = G_p.*(-2*pi*(ug.*(a^2).*d_u_du0 + vg.*(b^2).*d_v_du0));
dG_p_dv0 = G_p.*(-2*pi*(ug.*(a^2).*d_u_dv0 + vg.*(b^2).*d_v_dv0));
dG_m_du0 = G_m.*(-2*pi*(um.*(a^2).*d_u_du0 + vm.*(b^2).*d_v_du0));
dG_m_dv0 = G_m.*(-2*pi*(um.*(a^2).*d_u_dv0 + vm.*(b^2).*d_v_dv0));

dG_p_dtheta = G_p.*2*pi.*ug.*vg.*(a^2-b^2);
dG_m_dtheta = G_m.*2*pi.*um.*vm.*(a^2-b^2);

var_names = {'u0','v0','theta','a','b'};

% dF_vec = [];
for ii = 1:length(var_names)
    var_name = var_names{ii};
    dG_p_ds = ['dG_p_d',var_name];
    dG_m_ds = ['dG_m_d',var_name];
%     eval(['dq_ds = 2.*(G_p.*',dG_p_ds,'+ G_m.*',dG_m_ds,...
%           ' + cos(2*P).*(G_m.*',dG_p_ds,' +G_p.*',dG_m_ds,'));']);
     eval(['dq_ds = 2.*(G_p.*',dG_p_ds,'+ G_m.*',dG_m_ds,');']);
  %         '+(G_m.*',dG_p_ds,' +G_p.*',dG_m_ds,'));']);
      
    dF_ds_temp = vdiff.*K.*(q.^-0.5).*dq_ds;
    
    dF_ds_temp(isnan(dF_ds_temp)|isinf(dF_ds_temp))=0;
    
    dF_ds = sum(dF_ds_temp(:));
    
    dF_vec(ii) = dF_ds;
end

% df = [dF_dK,dF_vec,dF_dP]';
df = [dF_dK,dF_vec]';

f;
end


