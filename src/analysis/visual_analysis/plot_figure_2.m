%Author: Yosef Singer

% clear;
% close all;
%% Load data 
load('/path/to/fitted/model/Gabor/results.mat)'
real_data = load('/path/to/real/RFs.mat');
%set the seed to a reproducable number
rng(142);
% Fonts
FontName = 'Arial';
FSsm = 7; % small font size
FSmed = 12; % medium font size
FSlrg = 16; % medium font size
FSlabel = 22;

% Line widths
LWthin = 1; % thin lines

%Colours for model and real data
model_col = [120 120 120]./256;
data_col = 'red';
%Create figure
fig1 = figure(111);
set(fig1,'Position',[1000,1000,1400,1400])
clf;

%Plot the auditory STRFS
plot_aud_strfs_for_figure_2;

valid_real_idx = [1:7];
p = real_data.p(valid_real_idx,:,:,1:end-1);
n = real_data.n(valid_real_idx,:,:,1:end-1);
real_RF_size = real_data.settings.RF_size; 
real_clip_length = real_data.settings.clip_length;

numweights = size(weights,1)

[new_gabor_params, new_fitted_Gs, new_r2] = fixAliasedSFs(gabor_params, weights, fitted_Gs,r2);
[sfs,thets,X0,Y0,nx,ny,mask, best_r2, best_fitted_Gs, best_sigma_x, best_sigma_y] = getPopulationMeasures_fitted_best_t(new_gabor_params,new_r2,fitted_Gs);

original_weights = weights;
weights = weights(mask,:,:,:);
numweights = size(weights,1);

X0 = X0(mask);
Y0 = Y0(mask);
thets = thets(mask);
sfs = sfs(mask);
nx=nx(mask);
ny=ny(mask);

sep = assessSeperability(weights,0.5);
twoDstrfs = get2DSTRFs(X0,Y0,thets, weights);
tfs = getTemporalFreq(twoDstrfs);

nx = nx(sep);
ny = ny(sep);

%% Format real data
num_real_weights = size(p, 1);
real_thets = zeros(num_real_weights,1);
real_X0 = real_thets;
real_Y0 = real_thets;
real_strfs = get2DSTRFs(real_X0, real_Y0, real_thets,p-n);
real_tfs = getTemporalFreq(real_strfs);
real_sep = assessSeperability(p-n, 0.4);

selected_rf_ids = [59,165,98,40,47,95] 

for ii = 1:length(selected_rf_ids)
    id = selected_rf_ids(ii);
    selected_rfs(ii,:,:,:) = squeeze(weights(id,:,:,:));
    selected_twoDstrfs(ii,:,:) = squeeze(twoDstrfs(id,:,:));
end
%let's make sure that all units show leading excitation
flip_m_selected_ix = squeeze(sum(sum(selected_rfs(:,:,:,end),2),3))<0;
selected_rfs(flip_m_selected_ix,:,:,:) = -selected_rfs(flip_m_selected_ix,:,:,:);
selected_twoDstrfs(flip_m_selected_ix,:,:) = -selected_twoDstrfs(flip_m_selected_ix,:,:);

num_selected_rfs = length(selected_rf_ids);
mx_rf = max(abs(selected_rfs(:)));
mx_strf = max(abs(selected_twoDstrfs(:)));

%Now for the real data
real_sep_ix = find(real_sep==1);
real_insep_ix = find(real_sep==0);
%real_selected_seps = real_insep_ix(7);
real_selected_seps = real_sep_ix(1);
real_selected_inseps = real_insep_ix(1);
real_selected_rf_ids = [real_selected_seps real_selected_inseps];
% real_selected_rf_ids = [12,3];

for ii = 1:length(real_selected_rf_ids)
    id = real_selected_rf_ids(ii);
    real_selected_rfs(ii,:,:,:) = squeeze(p(id,:,:,:)-n(id,:,:,:));
    real_selected_twoDstrfs(ii,:,:) = squeeze(real_strfs(id,:,:));
end

flip_r_selected_ix = squeeze(sum(sum(real_selected_rfs(:,:,:,end),2),3))<0;
real_selected_rfs(flip_r_selected_ix,:,:,:) = -real_selected_rfs(flip_r_selected_ix,:,:,:);
real_selected_twoDstrfs(flip_r_selected_ix,:,:) = -real_selected_twoDstrfs(flip_r_selected_ix,:,:);

real_num_selected_rfs = size(real_selected_rfs,1);
real_mx_rf = max(abs(real_selected_rfs(:)));
real_mx_strf = max(abs(real_selected_twoDstrfs(:)));

%**************************************************************************
%****************************Plot selected RFs*****************************
%**************************************************************************
% Axis position
top_margin = 0.05;   % space above figure
bottom = 0.55; % space below figure
hspace = 0.013;% horizontal space between plot components
vspace = 0.013;% vertical space between plot components
width = 0.05;
height = 0.05;
left_margin= mstrf_ax(end).Position(1) + 0.25;

%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%**************************************************************************
%Now the real data
cnt = 0;
for jj = 1: real_num_selected_rfs
    cnt = cnt+1;
    for ii = 1:6
        id = ii+1;
        start_w = left_margin + (ii-1)*(width+hspace);
        start_h = 1- top_margin -height -(jj-1)*(height+vspace);
        pos = [start_w, start_h, width,height];
        RF_positions(ii,jj).pos = pos;
        rf_ax(ii,jj) = axes('position',pos);
        this_RF = squeeze(real_selected_rfs(cnt,:,:,id));
        imagesc(this_RF', 'Parent', rf_ax(ii,jj),[-real_mx_rf real_mx_rf]);
        axis off;
        axis square;
    end
    start_w = left_margin + (ii)*(width+1.35*hspace);
    pos = [start_w, start_h, width, height];
    twoDstrf_positions(jj).pos = pos;
    strf_ax(jj) = axes('position',pos);
    this_strf = squeeze(real_selected_twoDstrfs(cnt,:,:));
    imagesc(this_strf,'Parent', strf_ax(jj),[-real_mx_strf real_mx_strf]);
    axis square;
    strf_ax(jj).TickLength = [0 0];
    strf_ax(jj).XTickLabel = '';
    strf_ax(jj).YTickLabel = '';
end
% We want to plot 4 timesteps of each selected RF
cnt = 0;
for jj = real_num_selected_rfs+1: num_selected_rfs+real_num_selected_rfs
    cnt = cnt+1;
    for ii = 1:6
        id = ii+1;
        start_w = left_margin + (ii-1)*(width+hspace);
        start_h = 1- top_margin -height -(jj-1)*(height+vspace);
        pos = [start_w, start_h, width,height];
        RF_positions(ii,jj).pos = pos;
        rf_ax(ii,jj) = axes('position',pos);
        this_RF = squeeze(selected_rfs(cnt,:,:,id));
        imagesc(this_RF', 'Parent', rf_ax(ii,jj),[-mx_rf mx_rf]);
        colormap 'gray';
        axis off;
        axis square;
    end
    start_w = left_margin + (ii)*(width+1.35*hspace);
    pos = [start_w, start_h, width, height];
    twoDstrf_positions(jj).pos = pos;
    strf_ax(jj) = axes('position',pos);
    this_strf = squeeze(selected_twoDstrfs(cnt,:,:));
    mx_strf = max(abs(this_strf(:)));
    imagesc(this_strf,'Parent', strf_ax(jj),[-mx_strf mx_strf]);
    axis square;
    strf_ax(jj).TickLength = [0 0];
    strf_ax(jj).XTickLabel = '';
    strf_ax(jj).YTickLabel = '';
end
%% Add text to figure
%**************************************************************************
%**************************************************************************
%******Unit label***************
num_ttl = annotation('textbox');
temp = RF_positions(end,1);
h =  temp.pos(2) + height + 0.015;
% w = 0.035;
temp2 = RF_positions(1,1);
w = temp2.pos(1)-width;
num_ttl_pos = [w h 0.01 0.01];
set(num_ttl,'Position',num_ttl_pos);
num_ttl.String = 'Unit';
num_ttl.LineStyle = 'none';
num_ttl.FontSize = FSlrg;
num_ttl.FontName = FontName;

%Time label
time_ttl = annotation('textbox');
h = RF_positions(1,end).pos(2)-2*height;
w = w - 0.04;
time_ttl_pos = [w h 0.08 0.08];
set(time_ttl,'Position',time_ttl_pos);
time_ttl.String = 'Time (ms)';
time_ttl.LineStyle = 'none';
time_ttl.FontSize = FSlrg;
time_ttl.FontName = FontName;
for jj = 1:2
    set(rf_ax(1,jj).YLabel,'String',(num2roman((jj))));
    set(rf_ax(1,jj).YLabel,'Visible','on');
    set(rf_ax(1,jj).YLabel,'Rotation',0);
    set(rf_ax(1,jj).YLabel,'VerticalAlignment','middle');
    set(rf_ax(1,jj),'FontSize',FSlrg);
    set(rf_ax(1,jj),'FontName',FontName);
%     set(rf_ax(1,jj),'FontWeight','bold'); 
end

%Label the model units
cnt = 0;
for jj = 3:size(rf_ax,2)
    cnt = cnt+1;
    set(rf_ax(1,jj).YLabel,'String',(num2roman((cnt))));
    set(rf_ax(1,jj).YLabel,'Visible','on');
    set(rf_ax(1,jj).YLabel,'Rotation',0);
    set(rf_ax(1,jj).YLabel,'VerticalAlignment','middle');
    set(rf_ax(1,jj),'FontSize',FSlrg);
    set(rf_ax(1,jj),'FontName',FontName);
%     set(rf_ax(1,jj),'FontWeight','bold'); 
end
for ii = 1:size(rf_ax,1)
    set(rf_ax(ii,end).XLabel,'String',num2str(-40*(6-ii)));
    set(rf_ax(ii,end).XLabel,'Visible','on');
    set(rf_ax(ii,end),'FontSize',FSlrg);
    set(rf_ax(ii,end),'FontName',FontName);
    set(rf_ax(ii,end).XLabel,'Margin',1);
%     set(rf_ax(ii,end),'FontWeight','bold'); 
end

%**************************************************************************
%Set the time below each timestep
set(strf_ax(end),'FontSize',FSlrg);
set(strf_ax(end),'FontName',FontName);
set(strf_ax(end).YLabel,'Visible','on')
set(strf_ax(end).XLabel,'Visible','on')
set(strf_ax(end).YLabel,'String','Space')
set(strf_ax(end).XLabel,'String','Time')

%% Add colored box around real RFs
data_col = 'red';

border_pos = [rf_ax(1,2).Position(1)-0.5*hspace,...
              rf_ax(1,2).Position(2)-0.5*vspace,...
              6*width+ 6*hspace,...
              2*height+2*vspace];
border_ax = axes('position',border_pos);
border_ax.Color = 'None';
border_ax.TickLength=[0 0];
border_ax.Box = 'on';
border_ax.XTickLabel = [];
border_ax.YTickLabel = [];
border_ax.XColor = data_col;
border_ax.YColor = data_col;
border_ax.LineWidth =2;

% Set the colormaps of the RFs and of the strfs
for ii = 1:length(strf_ax)
    colormap(strf_ax(ii), redblue);
end
%Plot colorbar with scale;
cleft= twoDstrf_positions(end).pos(1);
ctop = twoDstrf_positions(end).pos(2);
cleft = cleft+1.5*hspace+width;
cbpos = [cleft, ctop, 0.001, height*2];


cbax1 = axes('position',cbpos);
cbax1.FontSize = FSmed;
cb1 = colorbar(cbax1);
cb_width = 0.01;
cb1.Position = [cleft+hspace,ctop,cb_width,height*2.3];
colormap(cb1,redblue)
axis off;
cb1.Ticks =[0 0.5 1];
cb1.TickLabels = {'-1', '0', '+1'};

cbax2 = axes('position',cbpos);
cbax2.FontSize = FSmed;
cleft2 = cleft - 0.25*cb_width;
cb2 = colorbar(cbax2);
cb2.Position = [cleft2,ctop,cb_width,height*2.3];

colormap(cb2,gray)
axis off;
cb2.Ticks =[0 0.5 1];
cb2.TickLabels = {};