# coding: utf-8
import numpy as np
from numpy import *
import math
import sys
import csv
C = 0.0
feature = [0,1,3,4,5]

def load_data(filename_X,filename_Y):
    data_x = genfromtxt(filename_X, delimiter=',')
    data_x = data_x[1:,:]
    data_y = genfromtxt(filename_Y)     
    data_y = data_y[1:]
    data_x = np.array(data_x)
    data_x = np.concatenate((data_x,data_x[:, feature] ** 2,data_x[:, feature] ** 3,data_x[:, feature] ** 4,
                np.log(data_x[:, feature] + 1e-20)), axis=1)
    return (data_x,np.array(data_y))

def load_testing_data(filename):
    data_x = genfromtxt(filename,delimiter=',')
    data_x = data_x[1:,:]
    data_x = np.array(data_x)
    data_x = np.concatenate((data_x,data_x[:, feature] ** 2,data_x[:, feature] ** 3,data_x[:, feature] ** 4,
                np.log(data_x[:, feature] + 1e-20)), axis=1)
    return (data_x)
    
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class Regression():
    def __init__ (self, l_rate, iteration):
        self.l_rate, self.iteration = l_rate, iteration

    def training(self,x,y):
        w = np.zeros(x.shape[1] + 1)
        self.mean, self.std = self.attribute(x)
        x = self.scale(x)
        x = self.bias(x)
        self.w_l_rate = 0.0
        self.l_rate_ori = self.l_rate
        for i in range(self.iteration):
            pred = sigmoid(np.dot(x,w))
            grad = -np.dot(x.T,(y - pred))
            #update l_rate
            self.w_l_rate = self.w_l_rate + grad **2
            self.l_rate = self.l_rate_ori / np.sqrt(self.w_l_rate)
            w = w - self.l_rate * (grad + C * np.sum(w))

            if i % 100 == 0:
                loss = self.lossfunc(w,x,y)
                print ('iteration: %d | Cost: %f' % (i,loss))
        return w

    def lossfunc(self,w,x,y):
        pred = sigmoid(np.dot(x,w))
        return (-np.mean(y * np.log(pred + 1e-20) + (1-y) * np.log(1 - pred + 1e-20)))

    def attribute(self, x):
        return np.mean(x,axis=0), np.std(x, axis=0)

    def scale(self,x):
        return (x - self.mean) / (self.std + 1e-20)

    def bias(self, x):
        return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

def attribute(x):
    return np.mean(x,axis=0), np.std(x, axis=0)

def scale(x,mean,std):
    return (x - mean) / (std + 1e-20)

def bias(x):
    return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

if __name__=='__main__':
    x,y = load_data(sys.argv[1],sys.argv[2])
    print(x.shape)
    test_data = load_testing_data(sys.argv[3])
    mean, std = attribute(test_data)
    test_data = scale(test_data,mean,std)
    test_data = bias(test_data)
    #adjustment here
    k = Regression(0.05,4000).training(x,y)
    ans = 0
    with open(sys.argv[4],"w+") as result:
        result.write("id,label\n")
        for i in range(test_data.shape[0]):
            temp = np.dot(k,test_data[i])
            if sigmoid(temp) > 0.5:
                ans = 1
            else:
                ans = 0
            result.write(str(i+1)+","+str(ans)+"\n")