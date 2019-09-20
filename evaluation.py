#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
import math
import seaborn as sns
import matplotlib.pyplot as plt
from pre_process import *

def evaluation(prediction: np.ndarray, groundtruth: np.ndarray):
    # sanity check
    if len(prediction) != len(groundtruth):
        raise TypeError
    
    tn,fp,fn,tp = 0,0,0,0 #true negative, false positive, false negative, true positive
    
    for i in range(len(prediction)):
        if prediction[i] == 0 and groundtruth[i] == 0:
            tn += 1
        if prediction[i] == 1 and groundtruth[i] == 0:
            fp += 1
        if prediction[i] == 0 and groundtruth[i] == 1:
            fn += 1
        if prediction[i] == 1 and groundtruth[i] == 1:
            tp += 1
    return tn,fp,fn,tp


################ This is the function to call for "Accuracy"############
def accuracy(prediction: np.ndarray, groundtruth: np.ndarray):
    tn,fp,fn,tp = evaluation(prediction,groundtruth)
    return 1.0*(tp+tn)/(tp+tn+fp+fn)

######## Confusion Matrix ##############
def confusion_m(prediction: np.ndarray, groundtruth: np.ndarray):
    tn,fp,fn,tp = evaluation(prediction,groundtruth)
    confusion_matrix = [[tp, fp],[fn,tn]]
    return confusion_matrix

def merge_chunks(data_split,indices):
    indices = list(indices).sort()
    if len([indices]) < 2:
        return data_split[0]
    data_merged = data_split[indices[0]]
    indices.remove(indices[0]) #remove the first element so that it does not get re-merged
    for i in indices:
        data_merged = np.concatenate(data_merged,data_split[i],axis=0)
        
    return data_merged
        


def cross_validation(model, data: np.ndarray, k: int):
    np.random.shuffle(data)
    data_split = np.array_split(data,k)
    indices = set(range(k)) # a set containing 0 to k-1
    acc_list = [] # the list containing all the output accuracies by k folds
    for fold in range(k):
        # merge the numpy arrays except for the validation set for training
        other_indices = indices - set([fold])
        training_set = merge_chunks(data_split,other_indices)
        test_set = data_split[fold]
        x_train = training_set[:,:-1]
        y_train = training_set[:,-1]
        x_test = test_set[:,:-1]
        y_test = test_set[:,-1]
        
        model.fit(x_train,y_train)
        y_prediction = model.predict(x_test)
        
        acc_list.append(accuracy(y_prediction,y_test))
    return sum(acc_list) / len(acc_list)


#  # Testing Cross-Validation by using scikit learn packages (which should be commented out before submission)
# from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression(penalty='l2')
# print(cross_validation(clf,process_wines(),5))
#
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# print(cross_validation(clf,process_wines(),5))
#
# from sklearn.svm import SVC
# clf = SVC(kernel='linear')
# print(cross_validation(clf,process_wines(),5))

