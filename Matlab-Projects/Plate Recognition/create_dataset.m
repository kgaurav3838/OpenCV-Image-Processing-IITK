directory = dir('train');

fileNames = {directory.name};
fileNames = fileNames(3:end);

images = cell(2, length(fileNames));

for i=1:length(fileNames)
    images(1,i) = {imread(['train', '\', cell2mat(fileNames(i))])};
    fileName = cell2mat(fileNames(i));
    images(2,i) = {fileName(1)};
end

save('imgData.mat', 'images');
