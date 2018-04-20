%Author: Yosef Singer
function dti_out = calculate_tdi(twoDstrfs)
    %calculate the tilt direction index as in:
    %https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1413500/
    
    dti_out = zeros(size(twoDstrfs,1),1);
    for ii=1:size(twoDstrfs,1)
        strf = squeeze(twoDstrfs(ii,:,:));
        strf_ft = abs(fft2(strf));
        abs_fft_shift = abs(fftshift(strf_ft));
%         imagesc(abs_fft_shift);
        [r, c, max_val] = max2(abs_fft_shift);
        middle_row = size(abs_fft_shift,1)/2 + 1;
        min_row =  2*middle_row-r;
        if min_row<20
            min_val = abs_fft_shift(min_row, c);
        else
            min_val=max_val
        end
        
        this_dti = (max_val-min_val)/(max_val+min_val);
        dti_out(ii) = this_dti;
    end
end