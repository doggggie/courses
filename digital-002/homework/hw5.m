% hw5 q7
file_noisy_image = 'digital_images_week5_quizzes_noisy.jpg';
file_orig_image = 'digital_images_week5_quizzes_original.jpg';

% 1. convert to uint8
im_noisy = im2double(imread(file_noisy_image));
im_orig = im2double(imread(file_orig_image));

% 2. 3x3 median filter
im_medfilt3 = im2double(medfilt2(im_noisy));

% 3. 3x3 median filter pass2
im_medfilt3_pass2 = im2double(medfilt2(im_medfilt3));

% 4. Compute the PSNR values between 
%    (a) the noise-free image and the noisy input
%    (b) the noise-free image and the 1-pass filtering output
%    (c) the noise-free image and the 2-pass filtering output
max_intensity = 255.0;
max_intensity = 1.0;
psnr_a = psnr(im_orig, im_noisy, max_intensity);
psnr_b = psnr(im_orig, im_medfilt3, max_intensity);
psnr_c = psnr(im_orig, im_medfilt3_pass2, max_intensity);
psnr_a, psnr_b, psnr_c

figure;
subplot(2,2,1)
imshow(im_noisy)
subplot(2,2,2)
imshow(im_medfilt3)
subplot(2,2,3)
imshow(im_medfilt3_pass2)
subplot(2,2,4)
imshow(im_orig)
