#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 14:31:35 2017
Digit Recognizer
@author: janny
"""

import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors  
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import tree

def ReadCSV(name):
    path='data/'
    csvfile=open(path+name,'r')
    csvfileData=csv.reader(csvfile)
    csvData=[]
    for line in csvfileData:
        csvData.append(line)
    csvData.pop(0)
    intData=np.array(csvData).astype(np.int32)
    return intData
        


def LoadTrainData():
    Data=ReadCSV("train.csv")  
    trainLable=Data[0:37000,0]
    trainData=Data[0:37000,1:]
    testLable=Data[37000:,0]
    trainData=Data[37000:,1:]
    trainData=np.where(trainData>0,1,0)
    testData=np.where(testData>0,1,0)
    return trainData,trainLable, testData,testLable  
    
def LoadTestData():
    Data=ReadCSV("test.csv")  
    testLable_sub=Data[:,0]
    testData_sub=Data[:,1:]
    testData_sub=np.where(testData_sub>0,1,0)
    return testData_sub,testLable_sub 

def saveResult(result,csvName):
    with open(csvName,'wb') as myFile:    
        myWriter=csv.writer(myFile)
        for i in result:
            tmp=[]
            tmp.append(i)
            myWriter.writerow(tmp)
            
def RandomForest(trainData,trainLable,testData,testLable):
    print ('RF is running') 
    md = RandomForestClassifier(n_estimators = 50, n_jobs = 4)
    md.fit(trainData, trainLable)
    rfPredict = md.predict(testData)
    rfScore=metrics.accuracy_score(testLable, rfPredict)
    return rfPredict,rfScore

def Knn(trainData,trainLable,testData,testLable):
    knn = neighbors.KNeighborsClassifier(n_neighbors=5,algorithm='kd_tree', weights='distance', p=3) 
    knn.fit(trainData, trainLable) 
    knnPredict = knn.predict(testData)
    knnscore=metrics.accuracy_score(testLable, knnPredict)
    saveResult(knnPredict,'knn_Result.csv')
    return knnPredict,knnscore

def SVM(trainData,trainLable,testData,testLable):
    clf = SVC(kernel='rbf', C=10)
    clf.fit(trainData, trainLable)
    svmPredict=clf.predict(testData)     
    svmScore=metrics.accuracy_score(testLable, svmPredict)
    return svmPredict,svmScore
    
def SvmClassify(trainData,trainLable,testData,testLable):    #adding PCA
    pca = PCA(n_components=0.8, whiten=True)
    train_x = pca.fit_transform(trainData)
    test_x = pca.transform(testData)
    svc = SVC(kernel='rbf', C=10)
    svc.fit(train_x, trainLable)
    svmPredict2 = svc.predict(test_x)
    svmScore2=metrics.accuracy_score(testLable, svmPredict)
    return svmScore2
    
def LogClassify():
    clf = LogisticRegression()
    clf.fit(trainData, trainLable)
    logPredict = clf.predict(testData)
    logScore=metrics.accuracy_score(testLable, logPredict)
    return logPredict,logScore
    

def treeClassify(trainData,trainLable,testData,testLable):
    clf = tree.DecisionTreeClassifier()
    clf.fit(trainData,trainLable)
    treePredict=clf.predict(testData)
    treeScore=metrics.accuracy_score(testLable, treePredict)
    return treePredict,treeScore


def DigitalRecognizer():
    trainData,trainLable, testData,testLable  =LoadTrainData()
    pca = PCA(n_components=0.8, whiten=True)
    train_x = pca.fit_transform(trainData)
    test_x = pca.transform(testData)
    svmPredict,svmScore=SVM(train_x,trainLable,test_x,testLable)
    treePredict,treeScore=treeClassify(train_x,trainLable,test_x,testLable)
    rfPredict,rfScore=RandomForest(train_x,trainLable,test_x,testLable)
    
    ensembleData=np.zeros(len(treePredict),5)
    for i in range(len(treePredict)):
        ensembleData[i,:]=knnPredict[i],svmPredict[i],logPredict[i],treePredict[i],rfPredict[i]
    testData_sub,testLable_sub   =LoadTestData()
    rfPredict,rfScore=RandomForest(ensembleData,testLable,testData_sub,testLable_sub)
    return rfPredict,rfScore

rfPredict,rfScore=DigitalRecognizer()


