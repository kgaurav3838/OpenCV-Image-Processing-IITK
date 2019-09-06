image = imread('coins.png');
figure,imshow(image);
bwCoins = imbinarize(image);
figure,imshow(bwCoins);
clearBwCoins = imfill(bwCoins, 'holes');
figure,imshow(clearBwCoins);

boundaries = bwboundaries(clear_bwCoins);
figure,imshow(image);
text(10, 10, ['Bulunan Nesne Sayısı: ', num2str(length(boundaries))], 'color', 'r');

hold on
for i=1:length(boundaries)
       boundary = boundaries{i};
       plot(boundary(:,2), boundary(:,1), 'b', 'LineWidth', 2);
end

[boundaries, numOfElements] = bwlabel(clear_bwCoins);
properties = regionprops(boundaries, 'Area', 'Centroid');

for i=1:numOfElements
    center = properties(i).Centroid;
    x = center(1);
    y = center(2);
    
    if properties(i).Area < 2000
        text(x-10, y, '5C');
    else
        text(x-10, y, '10C');
    end
end

hold off
