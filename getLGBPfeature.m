function getLGBPfeature(M,LGBPPath)
LGBPData = {};
%parameter setting
scales = 5;
orientations = 8;
gaborRows = 39;
gaborColumns = 39;
downsamplingRows = 4;
downsamplingColumns = 4;
for i = 1:size(M,1)  %size(M,1): the count of images
     temp = M{i,1};% 
    %lionNames = M(1:end,2); %all the name of lions
    gaborArray = gaborFilterBank(scales,orientations,gaborRows,gaborColumns); % (40x542)x644
    featureVector = gaborFeatures(temp,gaborArray,downsamplingRows,downsamplingColumns); % (40x542)x644
    LGBPMat = [];
 %size(featureVector)
    k = 0;
    for j = 0:size(featureVector,1)/(scales*orientations):((scales*orientations - 1)*size(temp,1))
       % ((k+1)*(size(featureVector,1)/(scales*orientations)))% 40*(k+1)
        gaborData = featureVector((j+1):((k+1)*(size(featureVector,1)/(scales*orientations))),:);
        %u2 is for uniform LBP get 1x59 and  riu2 for uniform
        %rotation-invariant LBP get 1x10
         mapping=getmapping(8,'u2'); 
        H1=MainLBP(gaborData,1,8,mapping,'h'); 
        %size(H1); %1x59
        LGBPMat = [LGBPMat H1];
        k = k + 1;
    end
   
    LGBPData{i,1} = LGBPMat;
    LGBPData{i,2} =  M{i,2};
end
%LGBPDataPath = '.\LGBPLions\'
if ~exist(LGBPPath)
    mkdir(LGBPPath);
end
LGBP = strcat(LGBPPath,'LGBPFeature.mat');
save(LGBP,'LGBPData')


end