function getGLCMfeature(M, GLCMpath)
%for getting GLCM features
GLCMData = {};
for i = 1:size(M,1)
    A = M{i,1};
    [glcm,SI] = graycomatrix(A,'NumLevels',9,'GrayLimits',[]);
    glcm = glcm(:)';
  GLCMData{i,1} = glcm;
  GLCMData{i,2} = M{i,2};
end
if ~exist(GLCMpath)
    mkdir(GLCMpath);
end
GLCM = strcat(GLCMpath,'GLCMFeature.mat');
save(GLCM,'GLCMData')

end