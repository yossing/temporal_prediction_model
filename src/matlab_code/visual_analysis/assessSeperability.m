%Author: Yosef Singer
function sepmask = assessSeperability(weights, sep_thresh,disp)
if nargin<2
    sep_thresh = 0.5;
end
if nargin<3
    disp = 1;
end
[numweights, RF_width, RF_height, clip_length] =size(weights);
for ii = 1:numweights
    this_scomp = svd(reshape(weights(ii,:,:,:),RF_width*RF_height,clip_length));
    s_components(ii,:) = this_scomp;
    sep_old(ii) = sum(this_scomp>=sep_thresh*max(this_scomp));
    sep(ii) = (this_scomp(2)/this_scomp(1))>=sep_thresh;
end
sepmask = ~sep;
sepmask_old = ~(sep>1);

if sum(sepmask==sepmask_old)
    display('The seperability criterion has not changed')
end

if disp>0
% display(['Seperability criterion: Number of singular values >= ',num2str(sep_thresh,2),'*max_value']);
display(['Seperability criterion: Ratio of first two singular values >= ',num2str(sep_thresh,2)]);
display(['Number of space-time inseperable weights: ',int2str(sum(sepmask==0))]);
display(['Number of space-time seperable weights: ',int2str(sum(sepmask==1))]);
end
