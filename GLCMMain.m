clc;
clear;
%for GLCM features: human, humanglass, lions, race
%all mat data path
allMatPath = '..\perClassMat\';
matFormat = '*.mat';
allMatData = dir([allMatPath matFormat]);

%for human
 allHumanMatPath = strcat(allMatPath, allMatData(3).name); %allMatData(1).name when not add race
 load(allHumanMatPath); %M is all human mat data, including matrix and filename
 disp('for human GLCM data');
 humanGLCMPath = './GLCMHuman/';
 getGLCMfeature(M,humanGLCMPath);
 
%  
% %for humanGlass
allHumanGlassMatPath = strcat(allMatPath, allMatData(4).name);%allMatData(2).name when not add race
 load(allHumanGlassMatPath); %M is all humanGlass mat data, including matrix and filename
  disp('for human glass GLCM data');
  humanGlassGLCMPath = './GLCMHumanGlass/';
   getGLCMfeature(M,humanGlassGLCMPath);
%    
% %%for lions, because the image has black border, so cut it from (19:560,
% %%63:706), get 542x644
 allLionsMatPath = strcat(allMatPath, allMatData(5).name);%allMatData(3).name when not add race
 load(allLionsMatPath); %M is all lions mat data, including matrix and filename
disp('for lions GLCM data');
lionsGLCMPath = './GLCMLions/';
getGLCMfeature(M,lionsGLCMPath);

%%for race train
%for asian
allAsianTrainMatPath = strcat(allMatPath, allMatData(2).name);
 load(allAsianTrainMatPath); %M is all lions mat data, including matrix and filename
 disp('for asian train GLCM data');
asianTrainGLCMPath = './GLCMAsianTrain/';
getGLCMfeature(M,asianTrainGLCMPath);
%for white
allWhiteTrainMatPath = strcat(allMatPath, allMatData(7).name);
 load(allWhiteTrainMatPath); %M is all lions mat data, including matrix and filename
 disp('for asian train GLCM data');
whiteTrainGLCMPath = './GLCMWhiteTrain/';
getGLCMfeature(M,whiteTrainGLCMPath);

%%for race test
%for asian
allAsianTestMatPath = strcat(allMatPath, allMatData(1).name);
 load(allAsianTestMatPath); %M is all lions mat data, including matrix and filename
 disp('for asian test GLCM data');
asianTestGLCMPath = './GLCMAsianTest/';
getGLCMfeature(M,asianTestGLCMPath);
%for white
allWhiteTestMatPath = strcat(allMatPath, allMatData(6).name);
 load(allWhiteTestMatPath); %M is all lions mat data, including matrix and filename
 disp('for white test GLCM data');
whiteTestGLCMPath = './GLCMWhiteTest/';
getGLCMfeature(M,whiteTestGLCMPath);
