import pre_process
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
import math
import seaborn as sns
import matplotlib.pyplot as plt


class LDA(object):
    def __init__(self, X):
        self.p = np.array([0, 0])
        self.u0 = np.zeros(len(X[0]))
        self.u1 = np.zeros(len(X[0]))
        self.sigma = np.zeros((len(X[0]), len(X[0])))

    def fit(self, X, y):
        zeros = np.count_nonzero(y == 0)
        ones = np.count_nonzero(y == 1)
        self.p[0] = float(zeros / (ones + zeros))
        self.p[1] = float(ones / (ones + zeros))
        for i in range(len(y)):
            if y[i] == 0:
                self.u0 += X[i]/ zeros
            elif y[i] == 1:
                self.u1 += X[i] / ones
        for i in range(len(y)):
            if y[i] == 0:
                self.sigma += (np.asmatrix((X[i] - self.u0)).transpose()).dot((np.asmatrix((X[i] - self.u0))))/(zeros + ones - 2)
            elif y[i] == 1:
                self.sigma += (np.asmatrix((X[i] - self.u1)).transpose()).dot((np.asmatrix((X[i] - self.u1))))/(zeros + ones - 2)

    def predict(self, X):
        y = []
        for data in X:
            logodds = math.log(self.p[1]/self.p[0])-0.5*np.dot(np.dot(self.u1, np.linalg.inv(self.sigma)),self.u1.transpose())+0.5*np.dot(np.dot(self.u0, np.linalg.inv(self.sigma)),self.u0.transpose())+np.dot(np.dot(data,np.linalg.inv(self.sigma)),(self.u1-self.u0).transpose())
            if logodds <= 0:
                y.append(0)
            else:
                y.append(1)
        return y




