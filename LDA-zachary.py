#!/usr/bin/env python
# coding: utf-8

import numpy as np

class LDA:
    def __init__(self):
        self.W0 = 0
        self.W = []

    def fit(self, X, y):
        ones = np.count_nonzero(y)
        zeros = y.size - ones
        p = [zeros/(zeros+ones), ones/(zeros+ones)]
        
        u0 = [0] * (np.size(X, 1))
        u1 = [0] * (np.size(X, 1))
        
        class0 = []
        class1 = []
        
        for index, data in enumerate(X):
            if y[index] == 0:
                u0=[sum(x) for x in zip(u0, data)]
                class0.append(data)
            if y[index] == 1:
                u1=[sum(x) for x in zip(u1, data)]
                class1.append(data)
                
        u0[:] = [x/zeros for x in u0]
        u1[:] = [x/ones for x in u1]

        class0[:] = [[x1 - x2 for (x1, x2) in zip(x, u0)] for x in class0]
        class1[:] = [[x1 - x2 for (x1, x2) in zip(x, u1)] for x in class1]
        
        matrix0 = np.zeros((len(class0[0]), len(class0[0])))
        matrix1 = np.zeros((len(class1[0]), len(class1[0])))
        
        for x in class0:
            matrix0=np.add(matrix0, (np.array(x).reshape((-1, 1)) @ np.array(x).reshape(1, -1)))
        for x in class1:
            matrix1=np.add(matrix1, (np.array(x).reshape((-1, 1)) @ np.array(x).reshape(1, -1)))
            
        matrix = np.divide(np.add(matrix0, matrix1), np.size(y)-2)
        self.W0 = np.log(p[1]/p[0]) + 1/2*np.dot(np.dot(np.array(u0), np.linalg.inv(matrix)), np.array(u0).reshape(-1,1)) - 1/2*np.dot(np.dot(np.array(u1), np.linalg.inv(matrix)), np.array(u1).reshape(-1,1))
        self.W = np.dot(np.linalg.inv(matrix), np.array([x1 - x2 for (x1, x2) in zip(u1, u0)]).reshape(-1,1))
        return
                        
    def predict(self, data):
        res = []
        for x in data:
            val = self.W0 + np.dot(x, self.W)
            if val > 0:
                res.append(1)
            else:
                res.append(0)
        return res


