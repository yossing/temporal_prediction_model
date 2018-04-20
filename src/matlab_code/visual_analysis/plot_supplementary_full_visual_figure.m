%Author: Yosef Singer

% clear;
% close all;

% Load data
load('/path/to/fitted/model/Gabor/results.mat)'
real_data = load('/path/to/real/RFs.mat');
load('/path/to/monkey/cat/mouse/nx_ny/data.mat');
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
%Create figure
fig1 = figure(106);
set(fig1,'Position',[1000,1000,1400,1400])
clf;

% Colors
% azure = hex2rgb('006fff');
% model_col = azure;
model_col = [120 120 120]./256;
data_col = [1,0,0];
valid_real_idx = [1:7];
p = real_data.p(valid_real_idx,:,:,1:end-1);
n = real_data.n(valid_real_idx,:,:,1:end-1);
real_RF_size = real_data.settings.RF_size; 
real_clip_length = real_data.settings.clip_length;

numweights = size(weights,1)
if size(r2,1) == 1
    r2 = r2';
end
[new_gabor_params, new_fitted_Gs, new_r2] = fixAliasedSFs_fitted_best_t(gabor_params, best_weights, fitted_Gs,r2);
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
%% Setup the axes to plot the selected RFs at each time point
selected_seps = [86,33,70];
selected_inseps = [2,55,62];
selected_rf_ids = [selected_seps selected_inseps];
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
left_margin = 0.035;  % space on LHS of figure, normalised units
top_margin = 0.05;   % space above figure
bottom = 0.55; % space below figure
hspace = 0.013;% horizontal space between plot components
vspace = 0.013;% vertical space between plot components
width = 0.05;
height = 0.05;
%% %Show best timestep of selected RFs
all_rf_left = left_margin;
all_rf_top = top_margin+0.05;
% all_rf_left = RF_positions(1,end).pos(1);
all_rf_space = 0.005;
num_disp = 54;
best_weights_masked = best_weights(mask,:,:);
all_rf_ix = randi(sum(mask),num_disp);
count = 1;
for ii = 1:6
    for jj =1:9
        
        start_w = all_rf_left + (ii-1)*(width+all_rf_space);
        start_h = 1- all_rf_top  -(jj-1)*(height+all_rf_space);
        pos = [start_w, start_h, width,height];
        all_rf_positions(ii,jj).pos = pos;
        all_rf_ax(ii,jj) = axes('position',pos);
        
        this_RF = squeeze(best_weights_masked(all_rf_ix(count),:,:));
        imagesc(this_RF, 'Parent', all_rf_ax(ii,jj),[-mx_rf mx_rf]);
        axis off;
        axis square;
        count = count+1;
    end
end
%%
left_margin= all_rf_positions(end,end).pos(1) + 0.18;
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

        %         colormap 'gray';
        axis off;
        axis square;
    end
    start_w = left_margin + (ii)*(width+1.35*hspace);
    pos = [start_w, start_h, width, height];

    twoDstrf_positions(jj).pos = pos;
    strf_ax(jj) = axes('position',pos);
    
    this_strf = squeeze(real_selected_twoDstrfs(cnt,:,:));
%     contour(this_strf,'Parent', strf_ax(jj));
    imagesc(this_strf,'Parent', strf_ax(jj),[-real_mx_strf real_mx_strf]);
    axis square;
%     axis off;
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

%Unit label
num_ttl = annotation('textbox');
temp = RF_positions(end,1);
h =  temp.pos(2) + height + 0.015;
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
%%
% Set the colormaps of the RFs and of the strfs
for ii = 1:length(strf_ax)
    colormap(strf_ax(ii), redblue);
end
%Plot colorbar with scale;
cleft= twoDstrf_positions(end).pos(1);
ctop = twoDstrf_positions(end).pos(2);
cleft = cleft+hspace+width;
cbpos = [cleft, ctop, 0.001, height*2];
cbax1 = axes('position',cbpos);
cbax1.FontSize = FSmed;
cb1 = colorbar(cbax1);
cb1.Position = [cleft+hspace,ctop,0.01,height*2.3];
colormap(cb1,redblue)
axis off;
cb1.Ticks =[0 0.5 1];
cb1.TickLabels = {'-1', '0', '+1'};
cbax2 = axes('position',cbpos);
cbax2.FontSize = FSmed;
cleft2 = all_rf_positions(end,end).pos(1) + 2*hspace+width;
cb2 = colorbar(cbax2);
cb2.Position = [cleft2,ctop,0.01,height*2.3];
colormap(cb2,gray)
axis off;
cb2.Ticks =[0 0.5 1];
cb2.TickLabels = {'-1', '0', '+1'};
%% Plot population measures
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Axis position

left_w = 0.1;
pop_space = 0.07;
pop_h = 0.25;
pop_w = 0.25;
start_w = left_w;
start_h = RF_positions(1,end).pos(2)-0.1- pop_h;
left_margin = 0.05;  % space on LHS of figure, normalised units
top_margin = 0.05;   % space above figure
bottom = 0.55; % space below figure
hspace = 0.013;% horizontal space between plot components
vspace = 0.013;% vertical space between plot components

start_w = left_margin + 0.25*pop_space;
pop_positions(1).pos = [start_w,start_h,pop_h,pop_w];
pop_ax(1) = axes('position',pop_positions(1).pos);

start_w = left_margin + 1.3*pop_space + pop_w;
pop_positions(2).pos = [start_w,start_h+pop_h/5,pop_h,pop_w];
pop_ax(2) = polaraxes('position',pop_positions(2).pos);

start_w = left_margin + 1.3*pop_space + pop_w;
pop_positions(4).pos = [start_w,start_h-pop_h/10,pop_w,pop_h/4];
pop_ax(4) = axes('position',pop_positions(4).pos);

start_w = left_margin + 2.6*pop_space + 2*pop_w;
pop_positions(3).pos = [start_w,start_h,pop_h,pop_w];
pop_ax(3) = axes('position',pop_positions(3).pos);

norm_real_strfs = real_strfs./max(abs(real_strfs(:)));
nrealstrfs = size(real_strfs,1);
r_pow = sum(squeeze(sum(norm_real_strfs.^2,2))./nrealstrfs,1);
r_pow = r_pow/(sum(r_pow(:)));
hold(pop_ax(1),'on');
plot(pop_ax(1),r_pow,'x','Color', data_col, 'linewidth', 2);
r_weights = reshape(p-n,size(p,1), 400,7);
norm_r_weights = r_weights./max(abs(r_weights(:)));
numweights = size(weights,1);
norm_twoDstrfs = twoDstrfs./max(abs(twoDstrfs(:)));
m_pow = sum(squeeze(sum(norm_twoDstrfs.^2,2))./numweights,1);
m_pow = m_pow/(sum(m_pow(:)));
plot(pop_ax(1),m_pow, 'o', 'Color', model_col, 'linewidth', 2);

pop_ax(1).XLim = [1 8];
pop_ax(1).XTick=[1:2:7];
power_lgd = legend(pop_ax(1),'Cat V1','Model','Location','Northwest');
power_lgd.Color = 'none';
power_lgd.Box = 'off';
plot(pop_ax(1),m_pow, 'Color', model_col, 'linewidth', 2);
plot(pop_ax(1),r_pow,'Color', data_col, 'linewidth', 2);
cnt = 1;
for ii = 1:2:7
    xlbls{cnt} = num2str(-(7-ii)*40);
    cnt= cnt+1;
end
pop_ax(1).XTickLabel=xlbls;
pop_ax(1).XLabel.String = 'Time (ms)';
pop_ax(1).YLabel.String = 'Proportion of power';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Orientation polar plot
plrplt =polarplot(pop_ax(2),thets,sfs,'linestyle', 'none', 'marker','o','color', model_col);
pop_ax(2).RLim= [0 0.25];
pop_ax(2).RAxisLocation = 0;
pop_ax(2).ThetaLim = [0 180];
%labelling the axes is not so straightforward
polar_thet_label = title(pop_ax(2),'Orientation (deg)');
polar_thet_label.FontWeight = 'normal';
polar_thet_label.FontSize = FSlrg;
polar_r_label = annotation('textbox');
polar_r_label_pos = [pop_positions(2).pos(1), pop_positions(2).pos(2)-0.045, 1, 0.08];
set(polar_r_label,'Position',polar_r_label_pos);
polar_r_label.String = 'Spatial frequency (cycles/pixel)';
polar_r_label.LineStyle = 'none';
polar_r_label.FontSize = FSlrg;
polar_r_label.FontName = FontName;
polar_r_label.FontWeight='normal';

%Add histogram showing distribution of spatial frequencies
histogram(rad2deg(thets(sfs>0.001)),'Parent',pop_ax(4), 'FaceColor', model_col)
xlim(pop_ax(4),[0,180]);
ylim(pop_ax(4),[0,250]);
pop_ax(4).XTick = [0,90,180];
pop_ax(4).YLabel.String = '# units';
pop_ax(4).XLabel.String = 'Orientation (deg)';

%% nx ny plot
%Model data
plot(abs(nx(ix)),abs(ny(ix)),'o','Parent',pop_ax(3),'linewidth',1, 'color', model_col);
hold(pop_ax(3),'on');
%Stryker data
mouse_col = [1,0.6,0.6];
plot(mouse_nx,mouse_ny,'*','Parent',pop_ax(3),'linewidth',1.25, 'color',mouse_col);
%Ringach data
monkey_col = [0.4,0,0];
plot(monkey_nx,monkey_ny,'+','Parent',pop_ax(3),'linewidth',1.5, 'color',monkey_col);
%Jones and Palmer data
cat_col = data_col;
plot(cat_nx,cat_ny,'x','Parent',pop_ax(3),'linewidth',1.5, 'color',cat_col);
xlim(pop_ax(3),[0,1.5]);
ylim(pop_ax(3),[0,1.5]);
nxny_lgd = legend(pop_ax(3),'Model','Mouse V1', 'Monkey V1', 'Cat V1', 'Location','Best');
nxny_lgd.Color = 'none';
nxny_lgd.Box = 'off';
pop_ax(3).XLabel.String = 'n_x';
pop_ax(3).YLabel.String = 'n_y';
pop_ax(3).XTick = [0,0.5,1,1.5,2];
%*****************************************************
for ii = 1:length(pop_ax)
    set(pop_ax(ii),'FontSize',FSlrg);
    set(pop_ax(ii),'FontName',FontName);
    set(pop_ax(ii),'Box','Off');
    set(pop_ax(ii),'Color','none');
%     set(pop_ax(ii,jj),'FontWeight','bold');
    
end
%*****************************************************
%Finally, label the plots
label_box_size = 0.01;
relevant_start_ws = [0.03,...
                     RF_positions(1,1).pos(1)-0.06,...
                     twoDstrf_positions(1,1).pos(1),...
                     pop_positions(1).pos(1)-0.04,...
                     pop_positions(2).pos(1)-0.04,...
                     pop_positions(2).pos(1)-0.04,...
                     pop_positions(3).pos(1)-0.04];                  
relevant_start_ws = relevant_start_ws-(2*label_box_size);

relevant_start_hs = [all_rf_positions(1,1).pos(2)+ 1.5*all_rf_positions(1,1).pos(4),...
                     RF_positions(1,1).pos(2)+ 1.5*RF_positions(1,1).pos(4),...
                     twoDstrf_positions(1,1).pos(2)+1.5*twoDstrf_positions(1,1).pos(4),...
                     pop_positions(1).pos(2) + pop_positions(1).pos(4),...
                     pop_positions(1).pos(2) + pop_positions(1).pos(4),...
                     pop_positions(4).pos(2) + pop_positions(4).pos(4),...
                     pop_positions(3).pos(2) + pop_positions(1).pos(4)];                     
relevant_start_hs = relevant_start_hs+(2*label_box_size);

labels = ['a','b','c','d','e','f','g'];
for ii = 1:length(labels)
    this_lbl = annotation('textbox');
    this_pos = [relevant_start_ws(ii), relevant_start_hs(ii),...
                label_box_size,label_box_size];
    set(this_lbl, 'Position',this_pos);
    this_lbl.String = labels(ii);
    this_lbl.LineStyle = 'none';
    this_lbl.LineStyle = 'none';
    this_lbl.FontSize = FSlabel;
    this_lbl.FontName = FontName;
    this_lbl.FontWeight='bold';
end
%% Correlation between temporal and spatial frequency tuning:
[rrr,ppp] = corrcoef(abs(tfs),abs(sfs));
display('The correlation between temporal and spatial frequency tuning is:')
display(['r^2 = ', num2str(rrr(2))]);
display(['p < ', num2str(ppp(2))]);