clc;
clear;

class_mat_output = './perClassMat/';
if ~exist(class_mat_output)
    mkdir(class_mat_output);
end
%%get grayimage matrix data
%for lions
lions_path = '../../experiment_database/lions/';
lions_output_path = './lionsAllMat/';
lions_format = '*.png';
lions_all_name = 'lionsAllMat.mat';
class = 'lion';
M_lions = getImageMatrix(lions_path, lions_format, lions_output_path,class_mat_output,lions_all_name,class);
% % 
% % %for human
human_path = '../../experiment_database/v1/';
human_output_path = './humanAllMat/';
human_format = '*.bmp';
human_all_name = 'humanAllMat.mat';
class = 'human';
M_human = getImageMatrix(human_path, human_format, human_output_path,class_mat_output,human_all_name,class);
% % 
% % %for human_glass
human_glass_path = '../../experiment_database/v4/';
human_glass_output_path = './humanGlassAllMat/';
human_glass_format = '*.jpg';
human_glass_all_name = 'humanGlassAllMat.mat';
class = 'humanGlass';
M_human_glass = getImageMatrix(human_glass_path, human_glass_format, human_glass_output_path,class_mat_output,human_glass_all_name,class);
% 


%%for race
%%for train
race_train_output_path = './raceTrain/';
race_train_input_path = '../../experiment_database/CASIA_Iris_Race/CASIA_Biosecure_OKI_train/';
% %for asian train
asian_train_path = strcat(race_train_input_path, 'asian/');
asian_train_output_path = strcat(race_train_output_path,'asianTrain/');
asian_train_formant = '*.bmp';
asian_train_all_name = 'asianTrainAllMat.mat';
class = 'asianTrain';
M_asian_train =  getImageMatrix(asian_train_path, asian_train_formant, asian_train_output_path,class_mat_output,asian_train_all_name,class);
%for white train
white_train_path = strcat(race_train_input_path, 'white/');
white_train_output_path = strcat(race_train_output_path,'whiteTrain/');
white_train_formant = '*.bmp';
white_train_all_name = 'whiteTrainAllMat.mat';
class = 'whiteTrain';
M_white_train =  getImageMatrix(white_train_path, white_train_formant, white_train_output_path,class_mat_output,white_train_all_name,class);
% %%for test
race_test_output_path = './raceTest/';
race_test_input_path = '../../experiment_database/CASIA_Iris_Race/CASIA_Biosecure_OKI_test/';
% %for asian
asian_test_path = strcat(race_test_input_path, 'asian/');
asian_test_output_path = strcat(race_test_output_path,'asianTest/');
asian_test_formant = '*.bmp';
asian_test_all_name = 'asianTestAllMat.mat';
class = 'asianTest';
M_asian_test =  getImageMatrix(asian_test_path, asian_test_formant, asian_test_output_path,class_mat_output,asian_test_all_name,class);
% %for white
white_test_path = strcat(race_test_input_path, 'white/');
white_test_output_path = strcat(race_test_output_path,'whiteTest/');
white_test_formant = '*.bmp';
white_test_all_name = 'whiteTestAllMat.mat';
class = 'whiteTest';
M_white_test =  getImageMatrix(white_test_path, white_test_formant, white_test_output_path,class_mat_output,white_test_all_name,class);



%%get LGBP data



