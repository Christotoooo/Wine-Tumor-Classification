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
        self.p[0] = np.count_nonzero(y == 0)
        self.p[1] = np.count_nonzero(y == 1)
        for i in range(len(y)):
            if y[i] == 0:
                self.u0 = np.add(X[i], self.u0)
            elif y[i] == 1:
                self.u1 = np.add(X[i], self.u1)
        self.u0 = np.divide(self.u0, self.p[0])
        self.u1 = np.divide(self.u1, self.p[1])
        for i in range(len(y)):
            if y[i] == 0:
                self.sigma = np.add(self.sigma, np.outer((X[i] - self.u0), (X[i] - self.u0).transpose()))
            elif y[i] == 1:
                self.sigma = np.add(self.sigma, np.outer((X[i] - self.u1), (X[i] - self.u1).transpose()))
        self.sigma = np.divide(self.sigma, (self.p[0] + self.p[1] - 2))

    def predict(self, X):
        y = []
        w0 = math.log(self.p[1]/self.p[0])
        w0 -= 0.5*np.dot(np.dot(self.u1.transpose(), np.linalg.inv(self.sigma)), self.u1)
        w0 += 0.5*np.dot(np.dot(self.u0.transpose(), np.linalg.inv(self.sigma)), self.u0)
        for data in X:
            w1 = np.dot(np.dot(data.transpose(), np.linalg.inv(self.sigma)), (self.u1 - self.u0))
            logodds = w0 + w1
            if logodds <= 0:
                y.append(0)
            else:
                y.append(1)
        return y
