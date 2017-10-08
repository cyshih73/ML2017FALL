# coding: utf-8
import numpy as np
from numpy import *
import math
import random
import sys
import csv
feature = [8,9]
# now /feature = [7,8,9,10,14,15,16,17]
# 5.49feature = [2,7,8,9,10,12,14,15,16,17]
#feature = [1,2,3,4,5,6,7,8,9]
feature_squ = [8,9]
# July : 2162 2521
def load_training_data(filename):
    data , x, y= [], [], []
    for i in range(18):
        data.append([])
    n_row = 0
    row = genfromtxt(filename, delimiter=',')
    row = np.nan_to_num(row)
    for r in row:
        if n_row != 0 and (n_row > 2521 or n_row < 2162):
            for i in range(3,27):
                data[(n_row-1)%18].append(float(r[i]))
        n_row += 1
    print(n_row)
    for i in range(11):     # increase data
        for j in range(471):
            x.append([])
            for t in feature:
                for s in range(9):
                    x[471*i+j].append(data[t][480*i+j+s])
            y.append(data[9][480*i+j+9])
    x = np.array(x)
    for t in feature_squ:
        x = np.c_[x,x[:,t]**2]
    x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
    return (x,np.array(y))

def load_testing_data(filename):
    row = genfromtxt(filename,delimiter=',')
    row = np.nan_to_num(row)
    data, n_row = [], 0
    for r in row:
        if n_row % 18 == 0:
            data.append([])
        if (n_row % 18) in feature:
            for i in range(2,11):
                data[n_row//18].append(float(r[i]))
        n_row += 1

    data = np.array(data)
    for t in feature_squ:
        data = np.c_[data,data[:,t]**2]
    return np.concatenate((np.ones((data.shape[0],1)),data), axis=1)

class Regression():
    def __init__ (self, l_rate, iteration):
        self.l_rate, self.iteration = l_rate, iteration

    def training(self,x,y,w):
        x_t = x.transpose()
        s_gra = np.zeros(len(x[0]))

        for i in range(self.iteration):
            hypo = np.dot(x,w)
            loss = hypo - y
            loss_a = np.sum(loss)   #deletable
            cost = np.sum(loss**2) / len(x)
            cost_a  = math.sqrt(cost)

            gra = np.dot(x_t,loss)
            s_gra += gra**2
            ada = np.sqrt(s_gra)
            w = w - self.l_rate * gra/ada

            if i % 10000 == 0:
                print ('iteration: %d | Cost: %f | Loss: %f ' % ( i,cost_a,loss_a))
        return w

if __name__=='__main__':

    x,y = load_training_data(sys.argv[1])
    test_data = load_testing_data(sys.argv[2])
    
    # weight,learning rate
    w = np.zeros(len(x[0]))
    print(x.shape)
    print(y.shape)
    print(test_data.shape)
    #adjustment here
    k = Regression(0.5,1000000).training(x,y,w)

    with open(sys.argv[3],"w+") as result:
        print("id,value",file=result)
        for i in range(len(test_data)):
            print("id_"+str(i)+","+str(np.dot(k,test_data[i])),file=result)

