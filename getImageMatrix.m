function M = getImageMatrix(path,fileFormat, allMatFilePath, perClassOutput, allName, className)


fileList = dir([path fileFormat]);
%filepath = strcat(path, fileList(1).name)

% fileList(1).name
% fileList(1)
%   imageName = fileList(1).name;
%      imageData = imread(filepath);
%      imshow(imageData)
if ~exist(allMatFilePath)
    mkdir(allMatFilePath);
end
M = {};
for i = 1: length(fileList)
    imageName = fileList(i).name;
    imagePath = strcat(path, imageName);
    imageData = imread(imagePath);
    if numel(size(imageData)) > 2
         imageGrayData = rgb2gray(imageData);
    else
        imageGrayData = imageData;
    end
    if strcmp(className, 'lion') % for lion, need cut
        imageGrayData = imageGrayData(19:560, 63:706);
    else
        imageGrayData = imageGrayData;
    end
    M{i,1} = imageGrayData;
    M{i,2} = imageName;
    imageMatName = strcat(allMatFilePath, imageName(1:end-4));
    imageMat = strcat(imageMatName, '.mat');
    save(imageMat, 'imageGrayData');
end
save(strcat(perClassOutput, allName), 'M')
end