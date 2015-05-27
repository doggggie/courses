function v = isnr(original, noisy, restored)

numer = norm(original - noisy, 'fro') ^ 2;
denom = norm(original - restored, 'fro') ^ 2;
v = 10 * log10(numer / denom);
end