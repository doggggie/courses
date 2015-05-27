function v = psnr(im1, im2, maxI)

[nx, ny] = size(im1); 
mse = sum(sum((im1 - im2) .* (im1 - im2))) / (nx * ny);
v = 10 * log10(maxI * maxI / mse);
end