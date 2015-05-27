% hw3 q8
im = imread('digital_images_week3_quizzes_original_quiz.jpg');

% 1. convert uint8 to double
imdble = im2double(im);

% 2. create 3x3 lowpass filter and apply to image
lowpass3 = ones(3,3)/9.0;
im3 = imfilter(imdble, lowpass3, 'replicate');

% 3. down-sample by removing every other row and col (2nd, 4th,...) 
%    Resulting image should be 240x180 pixles.
[nx, ny] = size(im3);
im_ds = im3(1:2:nx, 1:2:ny);

% 4. Create an all-zero array of 479x359. 
%    For every odd i in [1,359] and j in [1,479], 
%    set the value of (i,j) equal to the low-resolution
%    image value at ((i+1)/2,(j+1)/2). 
%    After this step you have inserted zeros into the low-resolution image. 
im_us = zeros(nx, ny);
for i = 1:2:nx
    for j = 1:2:ny
        im_us(i, j) = im_ds((i+1)/2, (j+1)/2);
    end
end


% 5. Convolve the result obtained from step 4 with a filter with coefficients 
%    [0.25,0.5,0.25;0.5,1,0.5;0.25,0.5,0.25] using imfilter. 
%    Provide imfilter with 2 args: the result from step 4 and the filter. 
%    It performs bilinear interpolation to obtain up-sampled image. 
filter4 = [0.25, 0.5, 0.25; 0.5, 1, 0.5; 0.25, 0.5, 0.25];
im_us = imfilter(im_us, filter4);

% 6. Compute the PSNR between the upsampled image obtained from step 5 and the original image.
mse = sum(sum((imdble - im_us) .* (imdble - im_us))) / (nx * ny);
psnr = 10 * log10(1.0 / mse);
return
%mse3 = sum(sum((imdble - im3) .* (imdble - im3))) / 256.0^2;
%psnr3 = 10 * log10(1.0 / mse3)
