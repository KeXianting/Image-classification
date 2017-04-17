clc;
clear;
%for LGBP data: human, humanglass, lions, race
tic
%all mat data path
allMatPath = '..\perClassMat\';
matFormat = '*.mat';
allMatData = dir([allMatPath matFormat]);

%for human
%  allHumanMatPath = strcat(allMatPath, allMatData(3).name); %allMatData(1).name when not add race
%  load(allHumanMatPath); %M is all human mat data, including matrix and filename
%  disp('for human LGBP data');
%  humanLGBPPath = './LGBPHuman/';
%  getLGBPfeature(M,humanLGBPPath);
%  
% %for humanGlass
% allHumanGlassMatPath = strcat(allMatPath, allMatData(4).name);%allMatData(2).name when not add race
%  load(allHumanGlassMatPath); %M is all humanGlass mat data, including matrix and filename
%   disp('for human glass LGBP data');
%   humanGlassLGBPPath = './LGBPHumanGlass/';
%    getLGBPfeature(M,humanGlassLGBPPath);
%    
% %%for lions, because the image has black border, so cut it from (19:560,
% %%63:706), get 542x644
%  allLionsMatPath = strcat(allMatPath, allMatData(5).name);%allMatData(3).name when not add race
%  load(allLionsMatPath); %M is all lions mat data, including matrix and filename
%    disp('for lions LGBP data');
% lionsLGBPPath = './LGBPLions/';
% getLGBPfeature(M,lionsLGBPPath);


%%for race train
%for asian
% allAsianTrainMatPath = strcat(allMatPath, allMatData(2).name);
%  load(allAsianTrainMatPath); %M is all lions mat data, including matrix and filename
%  disp('for asian train LGBP data');
% asianTrainLGBPPath = './LGBPAsianTrain/';
% getLGBPfeature(M,asianTrainLGBPPath);
% %for white
% allWhiteTrainMatPath = strcat(allMatPath, allMatData(7).name);
%  load(allWhiteTrainMatPath); %M is all lions mat data, including matrix and filename
%  disp('for asian train LGBP data');
% whiteTrainLGBPPath = './LGBPWhiteTrain/';
% getLGBPfeature(M,whiteTrainLGBPPath);

%%for race test
%for asian
allAsianTestMatPath = strcat(allMatPath, allMatData(1).name);
 load(allAsianTestMatPath); %M is all lions mat data, including matrix and filename
 disp('for asian test LGBP data');
asianTestLGBPPath = './LGBPAsianTest/';
getLGBPfeature(M,asianTestLGBPPath);
%for white
allWhiteTestMatPath = strcat(allMatPath, allMatData(6).name);
 load(allWhiteTestMatPath); %M is all lions mat data, including matrix and filename
 disp('for white test LGBP data');
whiteTestLGBPPath = './LGBPWhiteTest/';
getLGBPfeature(M,whiteTestLGBPPath);




%   6.5152e+03
time = toc










%size(featureVector);%(40x542)x644
% for j = 0:(scales*orientations)%:featureVector
%     gaborData = featureVector((j+1):(j+(scales*orientations)),:);
%     %一幅图像有40个gabor图像, 然后对每一个进行LBP,得到一个LGBP矩阵
%     size(gaborData)
% end



%%
% LGBPData = {};
% for i = 1:size(M,1)
%      lionTest = M{1,1};% size(lionTest,1)
%  %lionNames = M(1:end,2); %all the name of lions
%  
%     gaborArray = gaborFilterBank(scales,orientations,gaborRows,gaborColumns); % (40x542)x644
%     featureVector = gaborFeatures(lionTest,gaborArray,downsamplingRows,downsamplingColumns);
%     LGBPMat = [];
%     for j = 0:size(lionTest,1):((scales*orientations - 1)*size(lionTest,1))
%         gaborData = featureVector((j+1):(j+size(lionTest,1)),:);
%         %size(gaborData) 542x644
%          mapping=getmapping(8,'riu2'); 
%         H1=MainLBP(gaborData,1,8,mapping,'h'); 
%         %size(H1) 1x59
%         LGBPMat = [LGBPMat H1];
%     end
%     LGBPData{i,1} = LGBPMat;
%     LGBPData{i,2} =  M{1,2}
% end
% LGBPDataPath = '.\LGBPLions\'
% if ~exist(LGBPDataPath)
%     mkdir(LGBPDataPath);
% end
% save(LGBPDataPath,'LGBPData')












% for i = 1:size(M,1)
%     temp = M{i,1};
%     temp = temp(19:560, 63:706);
%     
% end


%%for Gabor


 
% size(imageGrayData)
% ii = imageGrayData(19:560, 63:706 )
% imshow(ii)
% imageGrayData = reshape(imageGrayData)
%imshow(imageGrayData)
% M = getLGBPfeature(input);