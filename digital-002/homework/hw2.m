% hw2 q7
im = imread('digital_images_week2_quizzes_lena.gif');
imdble = im2double(im);
lowpass3 = ones(3,3)/9.0;
im3 = imfilter(imdble, lowpass3, 'replicate');
mse3 = sum(sum((imdble - im3) .* (imdble - im3))) / 256.0^2;
psnr3 = 10 * log10(1.0 / mse3)
lowpass5 = ones(5,5)/25.0;
im5 = imfilter(imdble, lowpass5, 'replicate');
mse5 = sum(sum((imdble - im5) .* (imdble - im5))) / 256.0^2;
psnr5 = 10 * log10(1.0 / mse5)
