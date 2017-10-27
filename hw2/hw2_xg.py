# coding: utf-8
import numpy as np
from numpy import *
import math
import sys
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

feature = [0,1,3,4,5]

def load_data(filename_X,filename_Y):
    data_x = genfromtxt(filename_X, delimiter=',')
    data_x = data_x[1:,:]
    data_y = genfromtxt(filename_Y)
    data_y = data_y[1:]
    data_x = np.array(data_x)
    data_x = np.concatenate((data_x,data_x[:, feature] ** 2,data_x[:, feature] ** 3,data_x[:, feature] ** 4,
                np.log(data_x[:, feature] + 1) ** 2), axis=1)
    return (data_x,np.array(data_y))

def load_testing_data(filename):
    data_x = genfromtxt(filename,delimiter=',')
    data_x = data_x[1:,:]
    data_x = np.array(data_x)
    data_x = np.concatenate((data_x,data_x[:, feature] ** 2,data_x[:, feature] ** 3,data_x[:, feature] ** 4,
                np.log(data_x[:, feature] + 1) ** 2), axis=1)
    return (data_x)

def attribute(x):
    return np.mean(x,axis=0), np.std(x, axis=0)

def scale(x,mean,std):
    return (x - mean) / (std + 1e-20)

def bias(x):
    return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

if __name__=='__main__':
    x,y = load_data(sys.argv[1],sys.argv[2])
    mean,std = attribute(x)
    x = scale(x,mean,std)
    x = bias(x)
    test_data = load_testing_data(sys.argv[3])
    mean, std = attribute(test_data)
    test_data = scale(test_data,mean,std)
    test_data = bias(test_data)


    model = XGBClassifier(objective= 'binary:logistic',
        learning_rate =0.06,n_estimators=1100)
    model.fit(x,y)
    model.save_model("./yee.model")
    pred = model.predict(test_data)
    prediction = [round(value) for value in pred]
    id = 0
    with open(sys.argv[4],"w+") as result:
        print("id,label",file=result)
        for value in prediction:
            id += 1
            print(str(id)+","+str(int(value)),file=result)
