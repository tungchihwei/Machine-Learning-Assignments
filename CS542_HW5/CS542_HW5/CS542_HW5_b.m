img = imread('./Lenanoise.png');
img = int16(img);

[col, row] = size(img);
tmp = size(img);
before = img;

comb = @(x, N)(1 + mod(x-1, N));

flag = 1;    
count = 0;

while (flag && count < 1000) 
    count = count + 1;
    flag = 0;    
    for i = 1:row
        for j = 1:col
  
            row_i = img(j, i);
            
            no_change = (-1.0 * abs(row_i - before(j, i)) - (1.0 * (abs(row_i - img(comb(j - 1, col), i)) + abs(row_i - img(j, comb(i + 1, row))) + abs(row_i - img(comb(j+1,col), i)) + abs(row_i - img(j, comb(i - 1, row))))));
            row_i = min(255, row_i + 1);  
            pos_change = (-1.0 * abs(row_i - before(j, i)) - (1.0 * (abs(row_i - img(comb(j - 1, col), i)) + abs(row_i - img(j, comb(i + 1, row))) + abs(row_i - img(comb(j+1,col), i)) + abs(row_i - img(j, comb(i - 1, row))))));
            row_i = img(j, i);
            row_i = max(0, row_i - 1);
            neg_change = (-1.0 * abs(row_i - before(j, i)) - (1.0 * (abs(row_i - img(comb(j - 1, col), i)) + abs(row_i - img(j, comb(i + 1, row))) + abs(row_i - img(comb(j+1,col), i)) + abs(row_i - img(j, comb(i - 1, row))))));
            row_i = img(j, i);
            
            if pos_change > no_change
                flag = 1;
                img(j, i) = min(255, row_i + 1);
            end
            if neg_change > no_change
                flag = 1;
                img(j, i) = max(0, row_i - 1);
            end
        end
    end
end

imshow(uint8(img));

figure();
imshow(uint8(before));

correct = imread('./Lena.png');
figure();
imshow(uint8(correct));






