#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
import math
import seaborn as sns
import matplotlib.pyplot as plt
from pre_process import *

class Logistic(object):
    def __init__(self, learningR, Iterations):
        self.learningRate = learningR
        self.gradientDescentIterations = Iterations
        self.weights = []

    def sigmoid(self, gamma):
        if gamma < 0:
            return 1 - 1 / (1 + math.exp(gamma))
        else:
            return 1 / (1 + math.exp(-gamma))

    def addUps(self, MatrixW, MatrixX, Y):
        addOn = 0.0
        for num in range(0, len(MatrixX)):
            addOn = addOn + MatrixW[num] * MatrixX[num]
            addOn = Y - self.sigmoid(addOn)
            MatrixAdd = []
    
        for num in range(0, len(MatrixX)):
            MatrixAdd.append(MatrixX[num] * addOn)
        
        return MatrixAdd

    def fit(self, trainingDataMatrixX, trainingDataMatrixY):
        #X特征值的行数与列数
        numOfRow = len(trainingDataMatrixX)
        numOfColumn = len(trainingDataMatrixX[0])
        MatrixAddAll = []
        if len(self.weights) > 0:
            for num in range(0, numOfColumn + 1):
                MatrixAddAll.append(0)
        else:
            for num in range(0, numOfColumn + 1):
                self.weights.append(1)
                MatrixAddAll.append(0)
        
        for num in range(0, self.gradientDescentIterations):
            for numOne in range(0, numOfRow):
                MaX = trainingDataMatrixX[numOne]
                MaX = np.append([1],MaX)
                matrixAdd = self.addUps(self.weights, MaX, trainingDataMatrixY[numOne])
                for numTwo in range(0, numOfColumn + 1):
                    MatrixAddAll[numTwo] = matrixAdd[numTwo] + MatrixAddAll[numTwo]
            for numThree in range(0, numOfColumn + 1):
                MatrixAddAll[numThree] = self.learningRate * MatrixAddAll[numThree]
            for numFour in range(0, numOfColumn + 1):
                self.weights[numFour] = self.weights[numFour] + MatrixAddAll[numFour]
            for numFive in range(0, numOfColumn + 1):
                MatrixAddAll[numFive] = 0

        print(self.weights)
        return

    def predict(self, trainingDataMatrixX):
        outPutY = []
        for numOne in range(0, len(trainingDataMatrixX)):
            MatrixTemp = trainingDataMatrixX[numOne]
            sig = self.weights[0]
            for num in range(0, len(MatrixTemp)):
                sig = sig + self.weights[num + 1] * MatrixTemp[num]
            #if self.sigmoid(sig) >= 0.5:
            if sig >= 0.5:
                outPutY.append(1)
            else:
                outPutY.append(0)
        return outPutY