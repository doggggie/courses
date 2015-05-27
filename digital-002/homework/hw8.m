% hw8
filename = 'Cameraman256.bmp';

% 1. convert to uint8
im = im2uint8(imread(filename, 'bmp'));
[height, width] = size(im);

histo = zeros(256, 1);
for i = 1:height
    for j = 1:width
        histo(im(i,j)+1) = histo(im(i,j)+1) + 1;
    end
end

histo = histo / (height * width + 0.0);

entropy_e = 0.0;
entropy2 = 0.0;
for i = 1:256
    if histo(i) ~= 0.0
        entropy_e = entropy_e - histo(i) * log(histo(i));
        entropy2 = entropy2 - histo(i) * log2(histo(i));
    end
end


