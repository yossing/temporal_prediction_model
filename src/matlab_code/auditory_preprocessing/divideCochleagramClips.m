function [sound_clips, num_clips] = divideCochleagramClips(cg, clip_length, shift)

if ~exist('shift', 'var')
    shift = 0;
end

[numfreq, total_length] = size(cg);

total_length = total_length-shift;

num_clips =  floor(total_length/clip_length);

if shift == 0
    sound_clips = reshape(cg(:,1:num_clips*clip_length), [numfreq,clip_length, num_clips]);
    return
else 
%     sound_clips = zeros(numfreq,clip_length, num_clips);
    for clip_num = 1:num_clips
        start_frame = shift+1+(clip_num-1)*clip_length;
        sound_clips(:,:,clip_num) = cg(:,start_frame:start_frame+clip_length-1);
    end
    return
end