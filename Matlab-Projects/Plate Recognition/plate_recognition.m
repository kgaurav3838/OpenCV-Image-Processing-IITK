load('imgData.mat');

[file, filePath] = uigetfile({'*.jpg'; '*.bmp'}, 'Select an image');
file = [filePath, file];
image = imread(file);

[~,dimension] = size(image);
image = imresize(image, [300 500]);

if size(image, 3) == 3
    image = rgb2gray(image);
end

image = imbinarize(image);
figure,imshow(image);

reversedImage = ~image;
figure,imshow(reversedImage);

if dimension > 2000
    clearedImage = bwareaopen(reversedImage, 3500);
else
    clearedImage = bwareaopen(reversedImage, 3000);
end
figure,imshow(clearedImage);

plate = reversedImage - clearedImage;
plate = bwareaopen(plate, 250);
figure,imshow(plate);

[tags, Objects] = bwlabel(plate);
objectProperties = regionprops(tags,'BoundingBox');

hold on

pause(1);

for n=1:size(objectProperties, 1)
    rectangle('Position', objectProperties(n).BoundingBox, 'EdgeColor', 'g', 'LineWidth', 2);
end


hold off

figure

output = [];
t = [];

for n=1:Objects
    [r, c] = find(tags == n);
    char = plate(min(r):max(r), min(c):max(c));
    char = imresize(char, [42,24]);
    figure,imshow(char);
    
    x = [];
    noChar = size(images,2);
    
    for k=1:noChar
        y = corr2(images{1,k}, char);
        x = [x y];
    end
    
    t = [t max(x)];
    
    if max(x) > 0.4
        biggestIndex = find(x == max(x));
        outputChar = cell2mat(images(2, biggestIndex));
        output = [output, outputChar];
    end
end
