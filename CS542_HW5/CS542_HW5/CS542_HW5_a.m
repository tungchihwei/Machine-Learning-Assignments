img = imread('./Bayesnoise.png');
img = img(:,:,2);

img = int8(img);

[row, col] = size(img);
for i = 1:row
    for j = 1:col
        if img(i,j) < 127
            img(i,j) = -1;
        else
            img(i,j) = 1;
        end
    end
end
before = img;

img_size = size(before);
comb = @(x, N)(1 + mod(x-1, N));

stop=1;
count = 0;

while(stop) 
    count = count + 1;
    stop=0;    
    for i=1:img_size(2)
        for j=1:img_size(1)
    
            nonflip = img(j, i) * (-.01 - (5 * (img(j, comb(i+1, img_size(2))) + img(j, comb(i-1, img_size(2))) + img(comb(j+1, img_size(1)), i) + img(comb(j-1, img_size(1)), i))) - (3 * img(j,i)));
            flip = -1 * img(j, i) * (-.01 - (5 * (img(j, comb(i+1, img_size(2))) + img(j, comb(i-1, img_size(2))) + img(comb(j+1, img_size(1)), i) + img(comb(j-1, img_size(1)), i))) - (3 * img(j, i)));
            
            if flip < nonflip
                img(j, i) = -1 * img(j,i);
                stop = 1;
            end
        end
    end
end

after = (img + 1) / 2;

clean = imread('./Bayes.png');
clean = clean(:,:,2);
[row, col] = size(clean);
for i = 1:row
    for j = 1:col
        if clean(i,j) < 127
            clean(i,j) = -1;
        else
            clean(i,j) = 1;
        end
    end
end

[row, col] = size(after);
total = row*col;
correct = 0;
for i = 1:row
    for j = 1:col
        if clean(i,j) == after(i,j)
            correct = correct + 1;
        end
    end
end
accuracy = correct/total * 100;

after = uint8(after);
imshow(255 * after);

figure();
imshow(uint8(before) * 255);

fprintf('accuracy: %.2f%% \n', accuracy)




