clc;
clear;
LGBPPath = './LGBP/';
GLCMPath = './GLCM/';
LGBPMatrixDataPath = './matrixLGBP/';
GLCMMatrixDataPath = './matrixGLCM/';
if ~exist(LGBPMatrixDataPath)
    mkdir(LGBPMatrixDataPath)
end
if ~exist(GLCMMatrixDataPath)
     mkdir(GLCMMatrixDataPath)
end

LGBPAsianTest = strcat(LGBPPath,'LGBPAsianTest/LGBPFeature.mat');
LGBPAsianTrain = strcat(LGBPPath,'LGBPAsianTrain/LGBPFeature.mat');
LGBPHuman = strcat(LGBPPath,'LGBPHuman/LGBPFeature.mat');
LGBPHumanGlass = strcat(LGBPPath,'LGBPHumanGlass/LGBPFeature.mat');
LGBPLions = strcat(LGBPPath,'LGBPLions/LGBPFeature.mat');
LGBPWhiteTest =  strcat(LGBPPath,'LGBPWhiteTest/LGBPFeature.mat');
LGBPWhiteTrain =  strcat(LGBPPath,'LGBPWhiteTrain/LGBPFeature.mat');
%%
load(LGBPLions)
LGBPLions = [];
LGBPLionsName = {};
for i = 1:size(LGBPData,1)
    LGBPLions(i,:) = LGBPData{i,1};
    LGBPLionsName{i,1} = LGBPData{i,2};
end
LGBPMatrixDataPathLion = strcat(LGBPMatrixDataPath,'LGBPLions.mat');
LGBPMatrixDataPathLionName = strcat(LGBPMatrixDataPath,'LGBPLionsName.mat');
save(LGBPMatrixDataPathLion,'LGBPLions')
save(LGBPMatrixDataPathLionName,'LGBPLionsName')
%%
load(LGBPHuman)
LGBPHuman = [];
LGBPHumanName = {};
for i = 1:size(LGBPData,1)
    LGBPHuman(i,:) = LGBPData{i,1};
    LGBPHumanName{i,1} = LGBPData{i,2};
end
LGBPMatrixDataPathHuman = strcat(LGBPMatrixDataPath,'LGBPHuman.mat');
LGBPMatrixDataPathHumanName = strcat(LGBPMatrixDataPath,'LGBPHumanName.mat');
save(LGBPMatrixDataPathHuman,'LGBPHuman')
save(LGBPMatrixDataPathHumanName,'LGBPHumanName')
%%
load(LGBPHumanGlass)
LGBPHumanGlass = [];
LGBPHumanGlassName = {};
for i = 1:size(LGBPData,1)
    LGBPHumanGlass(i,:) = LGBPData{i,1};
   LGBPHumanGlassName{i,1} = LGBPData{i,2};
end
LGBPMatrixDataPathHumanGlass = strcat(LGBPMatrixDataPath,'LGBPHumanGlass.mat');
LGBPMatrixDataPathHumanGlassName = strcat(LGBPMatrixDataPath,'LGBPHumanGlassName.mat');
save(LGBPMatrixDataPathHumanGlass,'LGBPHumanGlass')
save(LGBPMatrixDataPathHumanGlassName,'LGBPHumanGlassName')

%%
load(LGBPAsianTrain)
LGBPAsianTrain = [];
LGBPAsianTrainName = {};
for i = 1:size(LGBPData,1)
    LGBPAsianTrain(i,:) = LGBPData{i,1};
   LGBPAsianTrainName{i,1} = LGBPData{i,2};
end
LGBPMatrixDataPathAsianTrain = strcat(LGBPMatrixDataPath,'LGBPAsianTrain.mat');
LGBPMatrixDataPathAsianTrainName = strcat(LGBPMatrixDataPath,'LGBPAsianTrainsName.mat');
save(LGBPMatrixDataPathAsianTrain,'LGBPAsianTrain')
save(LGBPMatrixDataPathAsianTrainName,'LGBPAsianTrainName')

%%
load(LGBPAsianTest)
LGBPAsianTest = [];
LGBPAsianTestName = {};
for i = 1:size(LGBPData,1)
    LGBPAsianTest(i,:) = LGBPData{i,1};
   LGBPAsianTestName{i,1} = LGBPData{i,2};
end
LGBPMatrixDataPathAsianTest = strcat(LGBPMatrixDataPath,'LGBPAsianTest.mat');
LGBPMatrixDataPathAsianTestName = strcat(LGBPMatrixDataPath,'LGBPAsianTestName.mat');
save(LGBPMatrixDataPathAsianTest,'LGBPAsianTest')
save(LGBPMatrixDataPathAsianTestName,'LGBPAsianTestName')
%%
load(LGBPWhiteTrain)
LGBPWhiteTrain = [];
LGBPWhiteTrainName = {};
for i = 1:size(LGBPData,1)
    LGBPWhiteTrain(i,:) = LGBPData{i,1};
   LGBPWhiteTrainName{i,1} = LGBPData{i,2};
end
LGBPMatrixDataPathWhiteTrain = strcat(LGBPMatrixDataPath,'LGBPWhiteTrain.mat');
LGBPMatrixDataPathWhiteTrainName = strcat(LGBPMatrixDataPath,'LGBPWhiteTrainName.mat');
save(LGBPMatrixDataPathWhiteTrain,'LGBPWhiteTrain')
save(LGBPMatrixDataPathWhiteTrainName,'LGBPWhiteTrainName')
%%
load(LGBPWhiteTest)
LGBPWhiteTest = [];
LGBPWhiteTestName = {};
for i = 1:size(LGBPData,1)
    LGBPWhiteTest(i,:) = LGBPData{i,1};
  LGBPWhiteTestName{i,1} = LGBPData{i,2};
end
LGBPMatrixDataPathLGBPWhiteTest = strcat(LGBPMatrixDataPath,'LGBPWhiteTest.mat');
LGBPMatrixDataPathLGBPWhiteTestName = strcat(LGBPMatrixDataPath,'LGBPWhiteTestName.mat');
save(LGBPMatrixDataPathLGBPWhiteTest,'LGBPWhiteTest')
save(LGBPMatrixDataPathLGBPWhiteTestName,'LGBPWhiteTestName')



% clc;
% clear;
% GLCMPath = './GLCM/';
% %GLCMPath = './GLCM/';
% GLCMMatrixDataPath = './matrixGLCM/';
% %GLCMMatrixDataPath = './matrixGLCM/';
% if ~exist(GLCMMatrixDataPath)
%     mkdir(GLCMMatrixDataPath)
% end
% % if ~exist(GLCMMatrixDataPath)
% %      mkdir(GLCMMatrixDataPath)
% % end
% 
% GLCMAsianTest = strcat(GLCMPath,'GLCMAsianTest/GLCMFeature.mat');
% GLCMAsianTrain = strcat(GLCMPath,'GLCMAsianTrain/GLCMFeature.mat');
% GLCMHuman = strcat(GLCMPath,'GLCMHuman/GLCMFeature.mat');
% GLCMHumanGlass = strcat(GLCMPath,'GLCMHumanGlass/GLCMFeature.mat');
% GLCMLions = strcat(GLCMPath,'GLCMLions/GLCMFeature.mat');
% GLCMWhiteTest =  strcat(GLCMPath,'GLCMWhiteTest/GLCMFeature.mat');
% GLCMWhiteTrain =  strcat(GLCMPath,'GLCMWhiteTrain/GLCMFeature.mat');
% %%
% load(GLCMLions)
% GLCMLions = [];
% GLCMLionsName = {};
% for i = 1:size(GLCMData,1)
%     GLCMLions(i,:) = GLCMData{i,1};
%     GLCMLionsName{i,1} = GLCMData{i,2};
% end
% GLCMMatrixDataPathLion = strcat(GLCMMatrixDataPath,'GLCMLions.mat');
% GLCMMatrixDataPathLionName = strcat(GLCMMatrixDataPath,'GLCMLionsName.mat');
% save(GLCMMatrixDataPathLion,'GLCMLions')
% save(GLCMMatrixDataPathLionName,'GLCMLionsName')
% %%
% load(GLCMHuman)
% GLCMHuman = [];
% GLCMHumanName = {};
% for i = 1:size(GLCMData,1)
%     GLCMHuman(i,:) = GLCMData{i,1};
%     GLCMHumanName{i,1} = GLCMData{i,2};
% end
% GLCMMatrixDataPathHuman = strcat(GLCMMatrixDataPath,'GLCMHuman.mat');
% GLCMMatrixDataPathHumanName = strcat(GLCMMatrixDataPath,'GLCMHumanName.mat');
% save(GLCMMatrixDataPathHuman,'GLCMHuman')
% save(GLCMMatrixDataPathHumanName,'GLCMHumanName')
% %%
% load(GLCMHumanGlass)
% GLCMHumanGlass = [];
% GLCMHumanGlassName = {};
% for i = 1:size(GLCMData,1)
%     GLCMHumanGlass(i,:) = GLCMData{i,1};
%    GLCMHumanGlassName{i,1} = GLCMData{i,2};
% end
% GLCMMatrixDataPathHumanGlass = strcat(GLCMMatrixDataPath,'GLCMHumanGlass.mat');
% GLCMMatrixDataPathHumanGlassName = strcat(GLCMMatrixDataPath,'GLCMHumanGlassName.mat');
% save(GLCMMatrixDataPathHumanGlass,'GLCMHumanGlass')
% save(GLCMMatrixDataPathHumanGlassName,'GLCMHumanGlassName')
% 
% %%
% load(GLCMAsianTrain)
% GLCMAsianTrain = [];
% GLCMAsianTrainName = {};
% for i = 1:size(GLCMData,1)
%     GLCMAsianTrain(i,:) = GLCMData{i,1};
%    GLCMAsianTrainName{i,1} = GLCMData{i,2};
% end
% GLCMMatrixDataPathAsianTrain = strcat(GLCMMatrixDataPath,'GLCMAsianTrain.mat');
% GLCMMatrixDataPathAsianTrainName = strcat(GLCMMatrixDataPath,'GLCMAsianTrainsName.mat');
% save(GLCMMatrixDataPathAsianTrain,'GLCMAsianTrain')
% save(GLCMMatrixDataPathAsianTrainName,'GLCMAsianTrainName')
% 
% %%
% load(GLCMAsianTest)
% GLCMAsianTest = [];
% GLCMAsianTestName = {};
% for i = 1:size(GLCMData,1)
%     GLCMAsianTest(i,:) = GLCMData{i,1};
%    GLCMAsianTestName{i,1} = GLCMData{i,2};
% end
% GLCMMatrixDataPathAsianTest = strcat(GLCMMatrixDataPath,'GLCMAsianTest.mat');
% GLCMMatrixDataPathAsianTestName = strcat(GLCMMatrixDataPath,'GLCMAsianTestName.mat');
% save(GLCMMatrixDataPathAsianTest,'GLCMAsianTest')
% save(GLCMMatrixDataPathAsianTestName,'GLCMAsianTestName')
% %%
% load(GLCMWhiteTrain)
% GLCMWhiteTrain = [];
% GLCMWhiteTrainName = {};
% for i = 1:size(GLCMData,1)
%     GLCMWhiteTrain(i,:) = GLCMData{i,1};
%    GLCMWhiteTrainName{i,1} = GLCMData{i,2};
% end
% GLCMMatrixDataPathWhiteTrain = strcat(GLCMMatrixDataPath,'GLCMWhiteTrain.mat');
% GLCMMatrixDataPathWhiteTrainName = strcat(GLCMMatrixDataPath,'GLCMWhiteTrainName.mat');
% save(GLCMMatrixDataPathWhiteTrain,'GLCMWhiteTrain')
% save(GLCMMatrixDataPathWhiteTrainName,'GLCMWhiteTrainName')
% %%
% load(GLCMWhiteTest)
% GLCMWhiteTest = [];
% GLCMWhiteTestName = {};
% for i = 1:size(GLCMData,1)
%     GLCMWhiteTest(i,:) = GLCMData{i,1};
%   GLCMWhiteTestName{i,1} = GLCMData{i,2};
% end
% GLCMMatrixDataPathGLCMWhiteTest = strcat(GLCMMatrixDataPath,'GLCMWhiteTest.mat');
% GLCMMatrixDataPathGLCMWhiteTestName = strcat(GLCMMatrixDataPath,'GLCMWhiteTestName.mat');
% save(GLCMMatrixDataPathGLCMWhiteTest,'GLCMWhiteTest')
% save(GLCMMatrixDataPathGLCMWhiteTestName,'GLCMWhiteTestName')




