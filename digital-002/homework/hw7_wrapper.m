clear all
close all

%% Simulate 1-D blur and noise
image_original = im2double(imread('Cameraman256.bmp', 'bmp'));
[H, W] = size(image_original);
blur_impulse = fspecial('motion', 7, 0);
image_blurred = imfilter(image_original, blur_impulse, 'conv', 'circular');
noise_power = 1e-4;
randn('seed', 1);
noise = sqrt(noise_power) * randn(H, W);
image_noisy = image_blurred + noise;

%figure; imshow(image_original, 'border', 'tight');
%figure; imshow(image_blurred, 'border', 'tight');
%figure; imshow(image_noisy, 'border', 'tight');
figure;
subplot(2,2,1); imshow(image_original);
subplot(2,2,2); imshow(image_blurred);
subplot(2,2,3); imshow(image_noisy);

%% CLS restoration
alpha = 1000;  % you should try different values of alpha
image_cls_restored = cls_restoration(image_noisy, blur_impulse, alpha);
%figure; imshow(image_cls_restored, 'border', 'tight');
subplot(2,2,4); imshow(image_cls_restored);

%% computation of ISNR
isnr_value = isnr(image_original, image_noisy, image_cls_restored);

alpha, isnr_value
