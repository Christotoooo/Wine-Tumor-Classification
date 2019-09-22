import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
import math
import seaborn as sns
import matplotlib.pyplot as plt

# useful global variables
wines_header, tumors_header, wines_global,tumors_global = [],[],[],[]

def normalize(x):
    if max(x) == min(x):
        return x
    return x / (max(x) - min(x))

def process_wines():
    with open("winequality-red.csv", 'r') as f:
        wines = list(csv.reader(f, delimiter=";"))
    global wines_header
    wines_header = np.array(wines[0])  # with label header
    wines = np.array(wines[1:], dtype=np.float)  # with label

    # clean malinformed values by deleting the rows they inhabit
    invalid_index = []
    for i in range(len(wines)):
        for number in wines[i]:
            if math.isnan(number):
                np.delete(wines, i, 0)

    # differentiate labels
    for i in tqdm(range(len(wines[:, -1]))):
        if wines[:, -1][i] >= 5:
            wines[:, -1][i] = 1
        else:
            wines[:, -1][i] = 0

    global wines_global
    wines_global = wines
    wines_x = wines[:, :-1]
    # wines_x = np.delete(wines_x,10,axis=1)
    # wines_x = np.delete(wines_x,9,axis=1)
    # wines_x = np.delete(wines_x,8,axis=1)
    # wines_x = np.delete(wines_x,7,axis=1)
    # wines_x = np.delete(wines_x,6,axis=1)
    # wines_x = np.delete(wines_x,5,axis=1)
    # wines_x = np.delete(wines_x,4,axis=1)
    # wines_x = np.delete(wines_x,3,axis=1)
    # wines_x = np.delete(wines_x,2,axis=1)
    # wines_x = np.delete(wines_x,1,axis=1)
    # wines_x = np.delete(wines_x, 0, axis=1)
    wines_y = wines[:, -1]

    for i in range(len(wines_x)):
        wines_x[i] = normalize(wines_x[i])

    return wines_x, wines_y


def process_tumors():
    with open("breast-cancer-wisconsin.data", 'r') as f:
        tumors = list(csv.reader(f, delimiter=";"))

    global tumors_header
    tumors_header = ["clump thickness", "cell size", "cell shape", "marginal adhesion", \
                     "single epithelial cell size", "number of bare nuclei", "bland chromatin", \
                     "number of normal nuclei", "mitosis", "label"]  # with label header but no IDs

    # highlight malinformed values
    invalid_index = []
    for i in tqdm(range(len(tumors))):
        tumors[i] = tumors[i][0].split(",")
        for j in range(len(tumors[i])):
            if tumors[i][j].isnumeric() == False:
                invalid_index.append(i)  # the whole row
        # differentiate labels
        if int(tumors[i][-1]) <= 2:
            tumors[i][-1] = '0'
        else:
            tumors[i][-1] = '1'

    # clean malinformed values by deleting the rows they inhabit
    invalid_index.sort(reverse=True)
    for i in invalid_index:
        tumors.remove(tumors[i])

    tumors = np.array(tumors[0:], dtype=np.float)
    tumors = tumors[:, 1:]

    global tumors_global
    tumors_global = tumors
    tumors_x = tumors[:,:-1]
    #tumors_x = np.delete(tumors_x,3,axis=1) #
    tumors_y = tumors[:,-1]

    for i in range(len(tumors_x)):
        tumors_x[i] = normalize(tumors_x[i])

    return tumors_x, tumors_y


def stats_wines():
    global wines_global
    wines = wines_global
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
    # sns.boxplot(data=pos_wines)
    # sns.boxplot(data=neg_wines)
    df = pd.DataFrame(wines, columns=wines_header)
    df.describe()
    # sns.pairplot(df) # density looks very suspicious
    # sns_plot = sns.pairplot(df)
    # sns_plot.savefig("wines_pairplot.png")
    df_pos = pd.DataFrame(pos_wines, columns=wines_header)
    df_pos.describe()
    df_neg = pd.DataFrame(neg_wines, columns=wines_header)
    df_neg.describe()


def stats_tumors():
    global tumors_global
    tumors = tumors_global
    pos_index = []
    neg_index = []
    # separate two classes
    for index in range(len(tumors)):
        if tumors[index][-1] == 1:
            pos_index.append(index)
        else:
            neg_index.append(index)
    pos_tumors = tumors[pos_index]
    neg_tumors = tumors[neg_index]

    df = pd.DataFrame(tumors, columns=tumors_header)
    df.describe()
    # sns.pairplot(df) # density looks very suspicious
    df_pos = pd.DataFrame(pos_tumors, columns=tumors_header)
    df_pos.describe()
    df_neg = pd.DataFrame(neg_tumors, columns=tumors_header)
    df_neg.describe()

# x,y = process_tumors()
# print(wines_global)
# print(x)
# print(y)
# for i in range(len(x)):
#     for j in range(len(x[0])):
#         if math.isnan(x[i][j]):
#             print(i,j)
