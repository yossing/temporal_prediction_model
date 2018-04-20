function sos = createBandpassFilter(min_freq, max_freq,Fs)

n = 3;
Wn = [(min_freq/2) (max_freq)]./(0.5*Fs);
ftype = 'bandpass';

% % Transfer Function design
% [b,a] = butter(n,Wn,ftype);      % This is an unstable filter

% Zero-Pole-Gain design
[z,p,k] = butter(n,Wn,ftype);
sos = zp2sos(z,p,k);