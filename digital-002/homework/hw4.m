% hw4 q8
frame1 = 'digital_images_week4_quizzes_frame_1.jpg';
frame2 = 'digital_images_week4_quizzes_frame_2.jpg';
blkpos_x = 65;
blkpos_y = 81;
blksz_x = 32;
blksz_y = 32;


% 1. convert uint8 to double
im = imread(frame1);
imdble1 = double(im);

im = imread(frame2);
imdble2 = double(im);

[nx, ny] = size(im); % 288x352

% 2. loop through all blocks in im1
best_x = 0;
best_y = 0;
best_err = Inf;

for i = 1 : (nx - blksz_x + 1)
    for j = 1 : (ny - blksz_y + 1)
        mae = 0.0;
        for ik = 1 : blksz_x
            for jk = 1 : blksz_y
                mae = mae + abs(imdble2(blkpos_x + ik - 1, blkpos_y + jk - 1) - ...
                                imdble1(i + ik - 1, j + jk - 1));
            end
        end
        mae = mae / (blksz_x * blksz_y);
        
%        if abs(i - 65) < 5 && abs(j - 81) < 5
%            i, j, mae
%        end
        
        if mae < best_err
            best_err = mae;
            best_x = i;
            best_y = j;
        end
    end
end

best_x, best_y, best_err
