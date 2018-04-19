%Author: Yosef Singer

function tfs = getTemporalFreq(strfs)
debug = false;
numstrfs = size(strfs,1);
%Videos are sampled at 25fps therefore:
max_freq = 25/2;
for ii=1:numstrfs
    this_strf = squeeze(strfs(ii,:,:))';
    if debug
        figure(8); subplot(2,2,1); imagesc(this_strf);
    end
    %Interpolate to get better resolution
%     this_strf = interp2(this_strf,2);
    
    if debug
    subplot(2,2,2); imagesc(this_strf);
    end
    %Start by taking the Fourier transform
    H_s = fftshift(fft2(this_strf));
    
    %Take only the real part
    abs_H_s = abs(H_s); 
    %Interpolate to get better resolution
    abs_H_s = interp2(abs_H_s,4);
    
    interp_width = size(abs_H_s,1);
    %Only use positive spatial frequencies
    interp_height = size(abs_H_s,2);
    pSpectrum = abs_H_s(:,round(0.5*interp_height):interp_height);
    
    if debug
        subplot(2,2,3); imagesc(abs_H_s);
        subplot(2,2,4); imagesc(pSpectrum);
        drawnow;
        pause(0.1);
    end
    [r,c,maxval] = max2(pSpectrum);
    tfs(ii) = (size(pSpectrum,1)/2+1-r).*max_freq/48.5;
end