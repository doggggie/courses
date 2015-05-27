% hw9
image_orig = 'HW9_Cameraman256.bmp';
image10 = 'HW9_Cameraman10.jpg';
image75 = 'HW9_Cameraman75.jpg';

max_intensity = 1.0;

im = im2double(imread(image_orig, 'bmp'));

imwrite(im, image75, 'jpg', 'quality', 75);
im75 = im2double(imread(image75));
psnr75 = psnr(im, im75, max_intensity);

imwrite(im, image10, 'jpg', 'quality', 10);
im10 = im2double(imread(image10));
psnr10 = psnr(im, im10, max_intensity);

