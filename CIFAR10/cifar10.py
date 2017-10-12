#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import urllib
import tarfile
import cPickle
import numpy as np

def downloadCIFAR10(downloadDir=os.path.join(os.getcwd(), 'cifar-10'), downloadUrl='http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'):
    if not os.path.exists(downloadDir):
        os.makedirs(downloadDir)

    targzPath = os.path.join(downloadDir, 'cifar-10.tar.gz')
    dataPath = os.path.join(downloadDir, 'cifar-10-batches-py')

    if not os.path.exists(targzPath) and not os.path.exists(dataPath):
        datasetUrl = urllib.URLopener()
        print('# Downloading CIFAR-10 into {}... '.format(downloadDir))
        datasetUrl.retrieve(downloadUrl, targzPath)

        with tarfile.open(targzPath) as tar:
            for item in tar:
                tar.extract(item, downloadDir)

    if os.path.exists(targzPath):
        os.remove(targzPath)

    return dataPath
    
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
        
    return dict

def oneHotVector(classIdx, numClasses):            
    v = np.zeros((len(classIdx), numClasses), dtype=np.int)  
    v[np.arange(0, len(v)), classIdx] = 1                
    return v

class cifar10:
    IMG_WIDTH = 32
    IMG_HEIGHT = 32
    IMG_CHANNELS = 3
    DOWNLOAD_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    
    dataPath = ''
    batchSize = 128
    trainData = np.array([])
    trainLabels = np.array([])
    testData = np.array([])
    testLabels = np.array([])
    currentIndexTrain = 0
    currentIndexTest = 0
    nTrainSamples = 0
    nTestSamples = 0
    classNames = []
    pTrain = []
    pTest = []
                  
    def __init__(self, batchSize=128, downloadDir=os.path.join(os.getcwd(), 'cifar-10')):
        self.batchSize = batchSize
        self.dataPath = downloadCIFAR10(downloadDir, self.DOWNLOAD_URL)
        self.loadCIFAR10()
        
        # changing from row major to column major order for our tf scripts
        self.trainData = self.trainData.reshape(-1, self.IMG_CHANNELS, self.IMG_WIDTH, self.IMG_WIDTH).transpose(0, 2, 3, 1).reshape(-1, self.IMG_CHANNELS * self.IMG_WIDTH * self.IMG_HEIGHT)
        self.testData = self.testData.reshape(-1, self.IMG_CHANNELS, self.IMG_WIDTH, self.IMG_WIDTH).transpose(0, 2, 3, 1).reshape(-1, self.IMG_CHANNELS * self.IMG_WIDTH * self.IMG_HEIGHT)
    
    def loadCIFAR10(self):
        trainFilenames = [os.path.join(self.dataPath, 'data_batch_%d' % i) for i in xrange(1, 6)]
        testFileName = os.path.join(self.dataPath, 'test_batch')
        self.classNames = unpickle(os.path.join(self.dataPath, 'batches.meta'))['label_names']                
         
        for filePath in trainFilenames:
            d = unpickle(filePath)            
             
            if self.trainData.size == 0:
               self.trainData = d['data']
               self.trainLabels = oneHotVector(d['labels'], 10)              
            else:
               self.trainData = np.concatenate((self.trainData, d['data']))
               self.trainLabels = np.concatenate((self.trainLabels, oneHotVector(d['labels'], 10)))            
                        
        d = unpickle(testFileName)
        self.testData = np.array(d['data'])
        self.testLabels = oneHotVector(d['labels'], 10)
        
        self.nTrainSamples = len(self.trainLabels)
        self.nTestSamples = len(self.testLabels)
        
        self.pTrain = np.random.permutation(self.nTrainSamples)              
        self.pTest = np.random.permutation(self.nTestSamples)                         
        
    def getTrainBatch(self, allowSmallerBatches=False):
        return self._getBatch('train', allowSmallerBatches)
                
    def getTestBatch(self, allowSmallerBatches=False):
        return self._getBatch('test', allowSmallerBatches)
    
    def _getBatch(self, dataSet, allowSmallerBatches=False):
        D = np.array([])
        L = np.array([])
        
        if dataSet == 'train':
            train = True
            test = False
        elif dataSet == 'test':
            train = False
            test = True
        else:
            raise ValueError('_getBatch: Unrecognised set: ' + dataSet)
                
        while True:       
            if train:
                r = range(self.currentIndexTrain, min(self.currentIndexTrain + self.batchSize - L.shape[0], self.nTrainSamples))                                            
                self.currentIndexTrain = r[-1] + 1 if r[-1] < self.nTrainSamples-1 else 0            
                (d, l) = (self.trainData[self.pTrain[r]][:], self.trainLabels[self.pTrain[r]][:])
            elif test:
                r = range(self.currentIndexTest, min(self.currentIndexTest + self.batchSize - L.shape[0], self.nTestSamples))                                            
                self.currentIndexTest = r[-1] + 1 if r[-1] < self.nTestSamples-1 else 0            
                (d, l) = (self.testData[self.pTest[r]][:], self.testLabels[self.pTest[r]][:])
                        
            if D.size == 0:
                D = d
                L = l
            else:                        
                D = np.concatenate((D, d))
                L = np.concatenate((L, l))
            
            if D.shape[0] == self.batchSize or allowSmallerBatches:
                break                                               
                
        return (D, L)
    
    def showImage(self, image):
        from matplotlib import pyplot as plt        
        plt.imshow(image.reshape((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS)), interpolation='nearest')
        plt.show()
    
    def reset(self):
        self.currentIndexTrain = 0
        self.currentIndexTest = 0
        self.pTrain = np.random.permutation(self.nTrainSamples)              
        self.pTest = np.random.permutation(self.nTestSamples)       
    
if __name__ == '__main__':
    cifar = cifar10(batchSize=128)            
    (trainImages, trainLabels) = cifar.getTrainBatch()            
    (testImages, testLabels) = cifar.getTestBatch()   
    
    (allTestImages, allTestLabels) = (cifar.testData, cifar.testLabels)
    #cifar.showImage(allTestImages[50])
