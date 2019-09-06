candyImage = imread('img.jpg');

grayImage = rgb2gray(candyImage);
se = strel('disk', 16);
bg = imclose(grayImage, se);
figure,imshow(bg)

candys = imsubtract(bg, grayImage);
figure,imshow(candys);

bwCandys = imbinarize(candys);
figure,imshow(bwCandys);

whiteCandys = imfill(bwCandys, 'holes');
figure,imshow(whiteCandys);

[labels, numItem] = bwlabel(whiteCandys, 4);

data = regionprops(labels, 'Eccentricity', 'Area', 'BoundingBox');
areas = [data.Area];
eccentricities = [data.Eccentricity];

minArea = mean(areas) - std(areas)*0.25;
faultyAreas = find(areas < minArea & eccentricities > 0.5);
faultyData = data(faultyAreas);

figure,imshow(candyImage);
hold on
for i=1:length(faultyData)
    boundary = rectangle('Position', faultyData(i).BoundingBox, 'LineWidth', 2);
    set(boundary, 'EdgeColor', [0 0 1]);
end
hold off
