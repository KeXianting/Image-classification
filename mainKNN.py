# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 19:46:01 2017

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 09:36:43 2017

@author: Administrator
"""
from numpy import * #导入numpy的函数库
import numpy as np

import scipy.io as scio
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import cross_validation
from matplotlib import colors
from sklearn.lda import LDA
import h5py as h5
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
from sklearn import neighbors as ng

def getTrainAndTestSet(path,trainNum, dataName):
	'get trainset and testset'
	Data = scio.loadmat(path)
	#lgbpLionsData = h5.File(path,'r')
	#print (lgbpLionsData)
	All = Data[dataName] #
	#获取矩阵的LGBP特征
	Features = All[0:shape(All)[0],:] #384x1
	
	#获取矩阵的最后一列,-1为倒数第一 shape(lgbpLionsAll)[0]得到行数,shape(lgbpLionsAll)[1]得到列数
	#lgbpLionsName = lgbpLionsAll[0:shape(lgbpLionsAll)[0],-1:shape(lgbpLionsAll)[1]]
	#lions train 
	trainSet = Features[0:trainNum,:] #200x1

	#lion test
	testSet = Features[trainNum:shape(All)[0],:] #183x1
	#lionsTestSetName = lgbpLionsName[trainNum+1:shape(lgbpLionsAll)[0],:]
	return trainSet,testSet


def LDAClassificationForIris(trainNum, *dataSet):
	'This function is for LDA classification'
	
	temp = np.concatenate((dataSet[2],dataSet[4],),axis=0) #for all human: human(200) + humanGlass(200); 
	trainSet = np.concatenate((dataSet[0],temp),axis=0) #lion + all human

	trainLabelOne = np.zeros((shape(dataSet[0])[0],1)) #first class label for lions
	trainLabelTwo = np.ones((shape(temp)[0],1)) #second class label for all human
	trainLabel = np.concatenate((trainLabelOne,trainLabelTwo),axis=0)

	testLabelOne = np.zeros((shape(dataSet[1])[0],1))
	testLabelTwo = np.ones((shape(dataSet[3])[0]+shape(dataSet[5])[0],1))
	testLabel = np.concatenate((testLabelOne, testLabelTwo),axis=0)
	#print (shape(testLabel)) #417x1
	testSetOne = np.array(dataSet[1])
	testSetTwo = np.concatenate((dataSet[3],dataSet[5]),axis=0)
	testSet = np.concatenate((testSetOne,testSetTwo),axis=0) #testSet : 417x2360
#	print ('++++++++++++++++++++')
#	print (shape(trainSet))
#	print (shape(trainLabel))
	print ('------------------------------')
	#print (trainSet.shape)
	#print (trainLabel.shape)
	clf = ng.KNeighborsClassifier(algorithm='kd_tree')
	clf.fit(trainSet, trainLabel)
	LDA(n_components=None, priors=None, shrinkage=None, solver='svd', store_covariance=False, tol=0.0001)
	print ('=========================The classification results are :=======================')
	classificationResult = clf.predict(testSet)
	#print (shape(classificationResult)) #1x417; 
	print (classificationResult)

	#save the classificationResult: the first column is true label, the second is classification label
	#testLabel.T: 转置; testLabel为1x417,classificationResult为417x1,维数不同,需要化为相同
	trueLabelAndClassifyLabel = np.concatenate((testLabel.T,[classificationResult]),axis=0)
	trueLabelAndClassifyLabel = trueLabelAndClassifyLabel.T
	#print (trueLabelAndClassifyLabel.shape)
	count = 0
	for i in range(1,shape(classificationResult)[0]):
		if testLabel[i] == classificationResult[i]:
			count = count + 1
	accurcay = count/classificationResult.shape[0]
	print ('======================The accurcay of LDA:==========================================')
	print (accurcay)

	print ('======================The scores:===============================================')
	weight = [0.0001 for i in range(classificationResult.shape[0])]
	for x in range(1,classificationResult.shape[0]):
		weight[x-1] = random.uniform(0,1)
	print(clf.score(testSet, testLabel,weight))

	print ('======================The Estimate probability=================================')
	estimate_pro = clf.predict_proba(testSet) # for get ROC
	print (estimate_pro)
	#print (estimate_pro.shape)

	#print ('======================Predicit confidence scores for samples:============================')
	#predicit_confidence = clf.decision_function(testSet)
	#print (predicit_confidence)
	#print (predicit_confidence.shape)
	#call ROC
	#yLabel = np.concatenate((trainLabel,testLabel),axis=0)
	#getROCCurve(testLabel, predicit_confidence)
	#交叉验证
	X = np.concatenate((trainSet,testSet),axis=0)
	Y = np.concatenate((trainLabel,testLabel),axis=0)
	kFold = cross_validation.KFold(len(X),6, shuffle=True)
	
	getROCCurve(clf,X, Y, kFold)




def LDAClassificationForRace(trainNum, *dataSet):
	'This function is for LDA classification'
	trainSet = np.concatenate((dataSet[0],dataSet[2]),axis=0) #lion + all human

	trainLabelOne = np.zeros((shape(dataSet[0])[0],1)) #first class label for lions
	trainLabelTwo = np.ones((shape(dataSet[2])[0],1)) #second class label for all human
	trainLabel = np.concatenate((trainLabelOne,trainLabelTwo),axis=0)

	testLabelOne = np.zeros((shape(dataSet[1])[0],1))
	testLabelTwo = np.ones((shape(dataSet[3])[0],1))
	testLabel = np.concatenate((testLabelOne, testLabelTwo),axis=0)

	testSetOne = dataSet[1]
	testSetTwo = dataSet[3]
	testSet = np.concatenate((testSetOne,testSetTwo),axis=0) #testSet : 417x2360

	clf = ng.KNeighborsClassifier(algorithm='kd_tree')
	clf.fit(trainSet, trainLabel)
	LDA(n_components=None, priors=None, shrinkage=None, solver='svd', store_covariance=False, tol=0.0001)
	print ('=========================The classification results are :=======================')
	classificationResult = clf.predict(testSet)
	#print (shape(classificationResult)) #417x1
	print (classificationResult)

	#save the classificationResult: the first column is true label, the second is classification label
	trueLabelAndClassifyLabel = np.concatenate((testLabel.T,[classificationResult]),axis=0)
	#print (trueLabelAndClassifyLabel.shape)
	count = 0
	for i in range(1,shape(classificationResult)[0]):
		if testLabel[i] == classificationResult[i]:
			count = count + 1
	accurcay = count/classificationResult.shape[0]
	print ('======================The accurcay of LDA:==========================================')
	print (accurcay)

	print ('======================The scores:===============================================')
	weight = [0.0001 for i in range(classificationResult.shape[0])]
	for x in range(1,classificationResult.shape[0]):
		weight[x-1] = random.uniform(0,1)
	print(clf.score(testSet, testLabel,weight))

	print ('======================The Estimate probability=================================')
	estimate_pro = clf.predict_proba(testSet)
	print (estimate_pro)
	print (estimate_pro.shape)

	#print ('======================Predicit confidence scores for samples:============================')
	#predicit_confidence = clf.decision_function(testSet)
	#print (predicit_confidence)
	#print (predicit_confidence.shape)
	#kFold = cross_validation.KFold(len(trainSet),6, shuffle=True)
	#getROCCurve(clf,trainSet, trainLabel, testSet)
	#交叉验证
	X = np.concatenate((trainSet,testSet),axis=0)
	Y = np.concatenate((trainLabel,testLabel),axis=0)
	kFold = cross_validation.KFold(len(trainSet),6, shuffle=True)
	getROCCurve(clf, X, Y, kFold)

    


def lgbpForIrisLDA():
	
	trainNum = 200
	#../表示上一级目录
	lgbpLions = '../../feature_extraction/matrixLGBP/LGBPLions.mat'
	lgbpHuman = '../../feature_extraction/matrixLGBP/LGBPHuman.mat'
	lgbpHumanGlass = '../../feature_extraction/matrixLGBP/LGBPHumanGlass.mat'
	#label
	lionLabel = 0;
	humanLabel = 1;
	humanGlassLabel = 1;
	#for lions
	(lionsTrainSet,lionsTestSet) = getTrainAndTestSet(lgbpLions,trainNum,'LGBPLions')

	#for human 
	(humanTrainSet,humanTestSet) = getTrainAndTestSet(lgbpHuman,trainNum,'LGBPHuman')
	#for humanglass
	(humanGlassTrainSet,humanGlassTestSet) = getTrainAndTestSet(lgbpHumanGlass,trainNum,'LGBPHumanGlass')
	#print (type(humanGlassTrainSet))
	#print (shape(humanGlassTrainSet))
	LDAClassificationForIris(trainNum, lionsTrainSet,lionsTestSet, \
		humanTrainSet, humanTestSet,\
		humanGlassTrainSet, humanGlassTestSet)

def lgbpForRaceLDA():
	
	
	#../表示上一级目录
	lgbpAsianTrainPath = '../../feature_extraction/matrixLGBP/LGBPAsianTrain.mat'
	lgbpAsianTestPath = '../../feature_extraction/matrixLGBP/LGBPAsianTest.mat'
	lgbpWhiteTrainPath = '../../feature_extraction/matrixLGBP/LGBPWhiteTrain.mat'
	lgbpWhiteTestPath = '../../feature_extraction/matrixLGBP/LGBPWhiteTest.mat'

	lgbpAsianTrainData = scio.loadmat(lgbpAsianTrainPath)
	lgbpAsianTrain = lgbpAsianTrainData['LGBPAsianTrain']

	lgbpAsianTestData = scio.loadmat(lgbpAsianTestPath)
	lgbpAsianTest = lgbpAsianTestData['LGBPAsianTest']

	lgbpWhiteTrainData = scio.loadmat(lgbpWhiteTrainPath)
	lgbpWhiteTrain = lgbpWhiteTrainData['LGBPWhiteTrain']

	lgbpWhiteTestData = scio.loadmat(lgbpWhiteTestPath)
	lgbpWhiteTest = lgbpWhiteTestData['LGBPWhiteTest']
	trainNum = 500
	#label
	asinLabel = 0; #asian
	whiteLabel = 1;
	LDAClassificationForRace(trainNum, lgbpAsianTrain,lgbpAsianTest, \
		lgbpWhiteTrain, lgbpWhiteTest)

	
	

def glcmForIrisLDA():
	
	trainNum = 200

	#../表示上一级目录
	glcmLions = '../../feature_extraction/matrixGLCM/GLCMLions.mat'
	glcmHuman = '../../feature_extraction/matrixGLCM/GLCMHuman.mat'
	glcmHumanGlass = '../../feature_extraction/matrixGLCM/GLCMHumanGlass.mat'
	
	#for lions
	(glcmLionsTrainSet,glcmLionsTestSet) = getTrainAndTestSet(glcmLions,trainNum,'GLCMLions')
	
	#for human 
	(glcmHumanTrainSet,glcmHumanTestSet) = getTrainAndTestSet(glcmHuman,trainNum,'GLCMHuman')
	#for humanglass
	(glcmHumanGlassTrainSet,glcmHumanGlassTestSet) = getTrainAndTestSet(glcmHumanGlass,trainNum,'GLCMHumanGlass')
	#print (type(humanGlassTrainSet))
	#print (shape(humanGlassTrainSet))
	LDAClassificationForIris(trainNum, glcmLionsTrainSet,glcmLionsTestSet, \
		glcmHumanTrainSet, glcmHumanTestSet,\
		glcmHumanGlassTrainSet, glcmHumanGlassTestSet)

def glcmForRaceLDA():
	#../表示上一级目录
	glcmAsianTrainPath = '../../feature_extraction/matrixGLCM/GLCMAsianTrain.mat'
	glcmAsianTestPath = '../../feature_extraction/matrixGLCM/GLCMAsianTest.mat'
	glcmWhiteTrainPath = '../../feature_extraction/matrixGLCM/GLCMWhiteTrain.mat'
	glcmWhiteTestPath = '../../feature_extraction/matrixGLCM/GLCMWhiteTest.mat'

	glcmAsianTrainData = scio.loadmat(glcmAsianTrainPath)
	glcmAsianTrain = glcmAsianTrainData['GLCMAsianTrain']

	glcmAsianTestData = scio.loadmat(glcmAsianTestPath)
	glcmAsianTest = glcmAsianTestData['GLCMAsianTest']

	glcmWhiteTrainData = scio.loadmat(glcmWhiteTrainPath)
	glcmWhiteTrain = glcmWhiteTrainData['GLCMWhiteTrain']

	glcmWhiteTestData = scio.loadmat(glcmWhiteTestPath)
	glcmWhiteTest = glcmWhiteTestData['GLCMWhiteTest']
	trainNum = 500
	#label
	asinLabel = 0; #asian
	whiteLabel = 1;
	LDAClassificationForRace(trainNum, glcmAsianTrain,glcmAsianTest, \
		glcmWhiteTrain, glcmWhiteTest)

def getROCCurve(clf, X, Y, kFold):
	print ('====================================get ROC ====================')
	#交叉验证
	mean_tpr = 0.0
	mean_fpr = np.linspace(0,1,100)
	
	for i, (trn,tst) in enumerate(kFold):
		#print (tst)
		proBas = clf.fit(X[trn], Y[trn]).predict_proba(X[tst])
		fpr,tpr,thresholds = roc_curve(Y[tst], proBas[:,1])
		mean_tpr += interp(mean_fpr,fpr,tpr)
		mean_tpr[0] = 0.0
		roc_auc = auc(fpr,tpr)
		#plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
		outVal = clf.score(X[tst], Y[tst])
		#print (outVal)
	#	plt.plot(fpr, tpr, lw=1, label='ROC')
	plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
	plt.plot(fpr, tpr, lw=1, color='#FF0000', label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

	mean_tpr /= len(kFold)
	mean_tpr[-1] = 1.0 						#坐标最后一个点为（1,1）
	mean_auc = auc(mean_fpr, mean_tpr)		#计算平均AUC值
	#画平均ROC曲线
	#print mean_fpr,len(mean_fpr)
	#print mean_tpr
	plt.plot(mean_fpr, mean_tpr,  '--',color='#0000FF',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=1)
	plt.xlim([-0.02, 1.02])
	plt.ylim([-0.02, 1.02])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()



if __name__ == "__main__":
	print ('This is for getting LGBP Iris LDA: "')
	print ('++++++++++++++++LGBP IRIS  +++++++++++++++++++++++')
	lgbpForIrisLDA()
	print ('This is for getting LGBP Race LDA ')
	print ('++++++++++++++++LGBP RACE  +++++++++++++++++++++++')
	lgbpForRaceLDA()

	print ('This is for getting GLCM Iris LDA:')
	print ('++++++++++++++++GLCM IRIS  +++++++++++++++++++++++')
	glcmForIrisLDA()

	print ('This is for getting GLCM Race LDA:')
	print ('++++++++++++++++GLCM RACE  +++++++++++++++++++++++')
	glcmForRaceLDA()





