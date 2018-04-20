%Author: Yosef Singer

% clear;
% close all;
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
% Colors
model_col = [120 120 120]./256;
data_col = 'red';

%Create figure
fig = figure(7);
set(fig,'Position',[1000,1000,1400,1400])
clf;
%% Load data 
load('/path/to/fitted/model/Gabor/results.mat)'
real_data = load('/path/to/real/RFs.mat');
load('/path/to/monkey/cat/mouse/nx_ny/data.mat');

valid_real_idx = [1:7];
p = real_data.p(valid_real_idx,:,:,1:end-1);
n = real_data.n(valid_real_idx,:,:,1:end-1);
real_RF_size = real_data.settings.RF_size; 
real_clip_length = real_data.settings.clip_length;

%% Format model data
numweights = size(weights,1)
if size(r2,1) == 1
    r2 = r2';
end
[new_gabor_params, new_fitted_Gs, new_r2] = fixAliasedSFs_fitted_best_t(gabor_params, best_weights, fitted_Gs,r2);
[sfs,thets,X0,Y0,nx,ny,mask, best_r2, best_fitted_Gs, best_sigma_x, best_sigma_y] = getPopulationMeasures_fitted_best_t(new_gabor_params,new_r2,fitted_Gs);
numweights = size(weights,1);
sep = assessSeperability(weights(:,:,:,:),0.5);
twoDstrfs = get2DSTRFs(X0,Y0,thets, weights);
tfs = getTemporalFreq(twoDstrfs);
X0 = X0(mask);
Y0 = Y0(mask);
nx = nx(sep&mask);
ny = ny(sep&mask);
thets = thets(mask);
sfs = sfs(mask);
tfs = tfs(mask);

%% Format real data
num_real_weights = size(p, 1);
real_thets = zeros(num_real_weights,1);
real_X0 = real_thets;
real_Y0 = real_thets;
real_strfs = get2DSTRFs(real_X0, real_Y0, real_thets,p-n);
real_tfs = getTemporalFreq(real_strfs);
real_sep = assessSeperability(p-n, 0.4);

%% Plot population measures
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Axis position
left_margin = 0.05;  % space on LHS of figure, normalised units
top_margin = 0.05;   % space above figure
bottom = 0.55; % space below figure
hspace = 0.013;% horizontal space between plot components
vspace = 0.013;% vertical space between plot components
pop_space = 0.1;
pop_h = 0.2;
pop_w = 0.2;
start_h = 1-top_margin-pop_h;
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

%% Temporal frequency distribution plot
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
ylim(pop_ax(4),[0,350]);
pop_ax(4).XTick = [0,90,180];
pop_ax(4).YLabel.String = '# units';
pop_ax(4).XLabel.String = 'Orientation (deg)';

%% nx ny plot
% start_h = 1-top_margin-3*pop_h - 2*pop_space;

%Model data
plot(abs(nx),abs(ny),'o','Parent',pop_ax(3),'linewidth',1, 'color', model_col);
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
end
%*****************************************************
%Finally, label the plots
label_box_size = 0.01;
relevant_start_ws = [pop_positions(1).pos(1)-0.05,...
                     pop_positions(2).pos(1)-0.05,...
                     pop_positions(2).pos(1)-0.05,...
                     pop_positions(3).pos(1)-0.05];
relevant_start_ws = relevant_start_ws-(2*label_box_size);
relevant_start_hs = [pop_positions(1).pos(2) + pop_positions(1).pos(4),...
                     pop_positions(1).pos(2) + pop_positions(1).pos(4),...
                     pop_positions(4).pos(2) + pop_positions(4).pos(4),...
                     pop_positions(3).pos(2) + pop_positions(3).pos(4)];                     
relevant_start_hs = relevant_start_hs+(2*label_box_size);
labels = ['a','b','c', 'd'];
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
