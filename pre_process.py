#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
import math
import seaborn as sns
import matplotlib.pyplot as plt

# useful global variables
wines_header = []
tumors_header = []


def process_wines():
    with open("winequality-red.csv", 'r') as f:
        wines = list(csv.reader(f, delimiter=";"))
    global wines_header 
    wines_header = np.array(wines[0])  # with label header
    wines = np.array(wines[1:], dtype=np.float) # with label
    
    # clean malinformed values by deleting the rows they inhabit
    invalid_index = []
    for i in range(len(wines)):
        for number in wines[i]:
            if math.isnan(number):
                np.delete(wines,i,0)
    
    # differentiate labels
    for i in tqdm(range(len(wines[:,-1]))):
        if wines[:,-1][i] >= 5:
            wines[:,-1][i] = 1
        else:
            wines[:,-1][i] = 0
            
    return wines

def process_tumors():
    with open("breast-cancer-wisconsin.data", 'r') as f:
        tumors = list(csv.reader(f, delimiter=";"))
    
    global tumors_header 
    tumors_header = ["clump thickness","cell size","cell shape","marginal adhesion",                 "single epithelial cell size","number of bare nuclei","bland chromatin",                 "number of normal nuclei","mitosis","label"] # with label header but no IDs
    
    # highlight malinformed values
    invalid_index = []
    for i in tqdm(range(len(tumors))):
        tumors[i] = tumors[i][0].split(",")
        for j in range(len(tumors[i])):
            if tumors[i][j].isnumeric() == False:
                invalid_index.append(i)    #the whole row
        #differentiate labels
        if int(tumors[i][-1]) <= 2:
            tumors[i][-1] = '0'
        else:
            tumors[i][-1] = '1'
    
    # clean malinformed values by deleting the rows they inhabit
    invalid_index.sort(reverse=True)
    for i in invalid_index:
        tumors.remove(tumors[i])
       
    tumors = np.array(tumors[0:],dtype=np.int)
    tumors = tumors[:,1:]
    return tumors

def stats_wines():
    wines = process_wines()
    pos_index = []
    neg_index = []
    # separate two classes
    for index in range(len(wines)):
        if wines[index][-1] == 1:
            pos_index.append(index)
        else:
            neg_index.append(index)
    pos_wines = wines[pos_index]
    neg_wines = wines[neg_index]
    #sns.boxplot(data=pos_wines)
    #sns.boxplot(data=neg_wines)
    df = pd.DataFrame(wines,columns=wines_header)
    df.describe()
    #sns.pairplot(df) # density looks very suspicious
    df_pos = pd.DataFrame(pos_wines,columns=wines_header)
    df_pos.describe()
    df_neg = pd.DataFrame(neg_wines,columns=wines_header)
    df_neg.describe()

def stats_tumors():
    tumors = process_tumors()
    pos_index = []
    neg_index = []
    #separate two classes
    for index in range(len(tumors)):
        if tumors[index][-1] == 1:
            pos_index.append(index)
        else:
            neg_index.append(index)
    pos_tumors = tumors[pos_index]
    neg_tumors = tumors[neg_index]
    
    df = pd.DataFrame(tumors,columns=tumors_header)
    df.describe()
    #sns.pairplot(df) # density looks very suspicious
    df_pos = pd.DataFrame(pos_tumors,columns=tumors_header)
    df_pos.describe()
    df_neg = pd.DataFrame(neg_tumors,columns=tumors_header)
    df_neg.describe()

