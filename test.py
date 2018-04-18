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

def cnn(X_train, y_train, X_test, y_test):
    from keras.models import Sequential  
    from keras.layers.core import Dense, Dropout, Activation  
    from keras.optimizers import RMSprop
    from keras.datasets import mnist  
    import numpy

    model = Sequential()  
    model.add(Dense(512, input_shape=(784,))) # 输入层，28*28=784  
    model.add(Activation('relu')) # 激活函数是tanh  
    model.add(Dropout(0.2)) # 采用50%的dropout

    model.add(Dense(512)) # 隐层节点500个  
    model.add(Activation('relu'))  
    model.add(Dropout(0.2))

    model.add(Dense(10)) # 输出结果是10个类别，所以维度是10  
    model.add(Activation('softmax')) # 最后一层用softmax

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy']) # 使用交叉熵作为loss函数

    #(X_train, y_train), (X_test, y_test) = mnist.load_data() # 使用Keras自带的mnist工具读取数据（第一次需要联网）

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2]) # 由于mist的输入数据维度是(num, 28, 28)，这里需要把后面的维度直接拼起来变成784维  
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])  
    Y_train = (numpy.arange(10) == y_train[:, None]).astype(int) # 参考上一篇文章，这里需要把index转换成一个one hot的矩阵  
    Y_test = (numpy.arange(10) == y_test[:, None]).astype(int)
    
    model.fit(X_train, Y_train, batch_size=200, nb_epoch=10, verbose=1, validation_data=(X_test,Y_test))  
    print ('test set' )
    model.evaluate(X_test, Y_test, batch_size=200, show_accuracy=True, verbose=1)  

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


