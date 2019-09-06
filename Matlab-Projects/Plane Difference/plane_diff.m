plane1 = imread('img1.jpg');
plane2 = imread('img2.jpg');
figure,imshow(plane1);
figure,imshow(plane2);

difference = imabsdiff(plane1, plane2);
figure,imshow(difference);

bwDifference = difference > 20;
clear_bwDifference = bwareaopen(bwDifference, 30);
figure,imshow(clear_bwDifference);

se = strel('square', 3);
mClear_bwDifference = imdilate(clear_bwDifference, se);

[boundaries, numOfObjects] = bwlabel(mClear_bwDifference);
stats = regionprops(boundaries, 'all');
centers = [stats.Centroid];

x = centers(1:2:end);
y = centers(2:2:end);

figure,imshow(plane2);
hold on
plot(x, y, 'yo', 'MarkerSize', 40, 'LineWidth', 4);
hold off
