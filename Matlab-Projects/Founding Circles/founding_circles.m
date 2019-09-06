image = imread('img1.png');
figure,imshow(image);

grayImage = rgb2gray(image);
bwImage = imbinarize(grayImage);
figure,imshow(bwImage);

clear_bwImage = bwareaopen(bwImage, 40);
figure,imshow(clear_bwImage);

clearFilled_bwImage = imfill(clear_bwImage, 'holes');
figure,imshow(clearFilled_bwImage);

[Labels, Boundaries] =  bwboundaries(clearFilled_bwImage);

figure,imshow(label2rgb(Boundaries, @jet, [.5 .5 .5]));

hold on
for i=1:length(Labels)
    boundary = Labels{i};
    plot(boundary(:,2), boundary(:,1), 'LineWidth', 2);
end

stats = regionprops(Boundaries, 'Area', 'Centroid');
areaThreshhold = 0.89;

for k=1:length(Labels)
    boundary = Labels{k};
    delta_sq = diff(boundary).^2;
    cevre = sum(sqrt(sum(delta_sq, 2)));
    alan = stats(k).Area;
    yOran = 4*pi*alan / cevre^2;
    
    if yOran > areaThreshhold
        merkez = stats(k).Centroid;
        plot(merkez(1), merkez(2), 'ko');
    end
end
