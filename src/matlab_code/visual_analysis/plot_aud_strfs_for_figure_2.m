%Author: Yosef Singer

%% Load data
real_filepath = '/path/to/real/auditory/strf/folder/'; 
r_numfreq = 32; r_n_h = 38; r_dt = 5; r_dF = 1/6; 
rstrfdir = dir([real_filepath '/n*.mat']);
for jj=1:numel(rstrfdir) 
    load([real_filepath '/' rstrfdir(jj).name])
    strftemp = modelfit(1).fit.kernel.k_fh;
    constant = modelfit(1).fit.kernel.c;
    a= strftemp(:,1:end-floor(10/r_dt));
    rstrfs(jj,:) = reshape(a, 1, size(a,1)*size(a,2));   
end 
numstrfs = size(rstrfs,1);
rstrfs = reshape(rstrfs, numstrfs, r_numfreq,r_n_h);

load('/path/to/model/auditory/strfs.mat');
theta{1} = network_params{1}';
theta{2} = network_params{3}';
Wih = theta{1};

l2_norm = sum(Wih.^2,2);
[sort_vals,sort_ix] = sort(l2_norm,'ascend');
sort_ix = sort_ix(sort_vals>0.01*max(sort_vals));
Wih = Wih(sort_ix, :); 
Who = Who(:,sort_ix);
weights = Wih;

numweights = size(weights,1);
if exist('settings','var')
	n_h = settings.n_h-settings.numpredict;
	numfreq = settings.numfreq;
    dF= 1/6;
    dt = settings.dt;
else
    n_h = 40;
	numfreq = 32;
    dF= 1/6;
    dt = 5;
end
weights = reshape(weights, numweights, numfreq,n_h);

%% General figure properties
% Fonts
FontName = 'Arial';
FSsm = 7; % small font size
FSmed = 12; % medium font size
FSlrg = 16; % large font size
FSlabel = 22;
% Line widths
LWthin = 1; % thin lines
% Colors
model_col = [120 120 120]./256;
data_col = [10 150 10]./256;
%% Figure 2a 
%**************************************************************************
%************************Draw good example strfs***************************
%**************************************************************************

% Select good example strfs:%
selected_model_rf_ids = [39, 151, 30,119,131, 129];

for ii = 1:length(selected_model_rf_ids)
    id = selected_model_rf_ids(ii);
    selected_model_rfs(ii,:,:) = squeeze(weights(id,:,:));
end
num_selected_rfs = length(selected_model_rf_ids);
mx_mrf = max(abs(selected_model_rfs(:)));

selected_real_rf_ids = [1, 36, 72,112,69,67];

for ii = 1:length(selected_model_rf_ids)
    id = selected_real_rf_ids(ii);
    selected_real_rfs(ii,:,:,:) = squeeze(rstrfs(id,:,:));
end
num_selected_rfs = length(selected_real_rf_ids);
mx_rrf = max(abs(selected_real_rfs(:)));
%%
% Axis position
clf
left_margin = 0.12;  % space on LHS of figure, normalised units
top_margin = 0.05;   % space above figure
bottom = 0.55; % space below figure
hspace = 0.011;% horizontal space between axes
vspavce = 0.11;
width = 0.075;
height = 0.075;
%plot the real and example strfs side by side
for jj = 1:num_selected_rfs
    start_w = left_margin;
    start_h = 1- top_margin -height -(jj-1)*(height+hspace);
    pos = [start_w, start_h, width,height];
    rstrf_positions(jj).pos = pos;
    rstrf_ax(jj) = axes('position',pos);
    this_strf = squeeze(selected_real_rfs(jj,:,19:end));
    this_mx = max(abs(this_strf(:)));
    imagesc(this_strf, 'Parent', rstrf_ax(jj),[-this_mx this_mx]);
    axis off;
    axis square;
    colormap redblue;
    rstrf_ax(jj).Box = 'On';
    rstrf_ax(jj).Visible = 'On';
    rstrf_ax(jj).XTick = '';
    rstrf_ax(jj).YTick = '';
    start_w = left_margin + (width+3*hspace);
    pos = [start_w, start_h, width, height];
    mstrf_positions(jj).pos = pos;
    mstrf_ax(jj) = axes('position',pos);
    this_strf = squeeze(selected_model_rfs(jj,:,21:end));
    sign_tlast = sign(sum(this_strf(:,end)));
    if sign_tlast <0
        this_strf = -this_strf;
    end
    this_mx = max(abs(this_strf(:)));
    imagesc(this_strf, 'Parent', mstrf_ax(jj),[-this_mx this_mx]);
    axis off;
    axis square;
    mstrf_ax(jj).Box = 'On';
    mstrf_ax(jj).Visible = 'On';
    mstrf_ax(jj).XTick = '';
    mstrf_ax(jj).YTick = '';
end
%Plot colorbar with scale;
cleft= mstrf_positions(end).pos(1);
ctop = mstrf_positions(end).pos(2);
cleft = cleft+width;
cbpos = [cleft, ctop, 0.001, height*2];
cbax = axes('position',cbpos);
cbax.FontSize = FSmed;
cb = colorbar(cbax);
cb_width = 0.01;
cb.Position = [cleft+hspace,ctop,cb_width,height*1];
axis off;
cb.Ticks =[0 0.5 1];
cb.TickLabels = {'-1', '0', '+1'};
colormap(cb, 'redblue');
%% Add text to figure
%**************************************************************************
%***************************Annotate example strfs*************************
%**************************************************************************
ttl(1) = title(rstrf_ax(1), 'Real');
ttl(2) = title(mstrf_ax(1), 'Model');
for ii = 1:length(ttl)
    set(ttl(ii),'FontSize',FSlrg);
    set(ttl(ii),'FontName',FontName);
%     set(ttl(ii),'FontWeight','bold'); 
end
set(rstrf_ax(end).XLabel,'string','Time (ms)');  
set(rstrf_ax(end).XLabel,'Visible','on');
set(rstrf_ax(end).XLabel,'Rotation',0);
set(rstrf_ax(end).YLabel,'string','Freq (kHz)');  
set(rstrf_ax(end).YLabel,'Visible','on');
set(rstrf_ax(end).YLabel,'Rotation',90);
set(rstrf_ax(end).YLabel,'VerticalAlignment','middle');
set(rstrf_ax(end),'FontSize',FSlrg);
set(rstrf_ax(end),'FontName',FontName);

rstrf_ax(end).YLabel.Units = 'characters';
rstrf_ax(end).YLabel.Position = [-10 3.7 0];
rstrf_ax(end).YTick = [0.5 16 32];
rstrf_ax(end).YTickLabel = {'18', '', '0.5'};
rstrf_ax(end).Visible ='on';
rstrf_ax(end).TickLength =[0.05 0.05];
rstrf_ax(end).XTick =[0.5 10 20.5];
rstrf_ax(end).XTickLabel = {'-100', '-50', '0'};

mstrf_ax(end).TickLength =[0.05 0.05];
mstrf_ax(end).XTick =[0.5 10 20.5];
mstrf_ax(end).XTickLabel = {'-100', '-50', '0'};
mstrf_ax(end).YTick = [];
mstrf_ax(end).XLabel.String = 'Time (ms)';
set(mstrf_ax(end),'FontSize',FSlrg);
set(mstrf_ax(end),'FontName',FontName);
%We want to add text independantly of axes a good way to do this is as
%suggested in http://uk.mathworks.com/matlabcentral/newsreader/view_thread/15277
num_size = 0.05;
for jj = 1:length(rstrf_positions)
    h = rstrf_positions(jj).pos(2)+0.15*height;
    w = 0.02;
    num_text = annotation('textbox');
    set(num_text,'LineStyle', 'none');
    set(num_text, 'Position',[w,h,num_size,num_size]);
    set(num_text,'String',num2roman(jj));
    set(num_text,'Visible','on');
    set(num_text,'FontSize',FSlrg);
    set(num_text,'FontName',FontName);
end

num_ttl = annotation('textbox');
temp = rstrf_positions(1);
h =  temp.pos(2) + height + 0.02;
w = w - 0.02;
num_ttl_pos = [w h 0.01 0.01];
set(num_ttl,'Position',num_ttl_pos);
num_ttl.String = 'Unit';
num_ttl.LineStyle = 'none';
num_ttl.FontSize = FSlrg;
num_ttl.FontName = FontName;

for ii = 1:length(rstrf_ax)
    colormap(rstrf_ax(ii), redblue);
end
for ii = 1:length(mstrf_ax)
    colormap(mstrf_ax(ii), redblue);
end