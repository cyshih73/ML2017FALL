# coding: utf-8
import numpy as np
from numpy import *
import math
import random
import sys
import csv

def load_training_data(filename):
    data , x, y= [], [], []
    for i in range(18):
        data.append([])
    n_row = 0
    row = genfromtxt(filename, delimiter=',')
    row = np.nan_to_num(row)
    for r in row:
        if n_row != 0:
            for i in range(3,27):
                data[(n_row-1)%18].append(float(r[i]))
        n_row += 1
    for i in range(12):     # increase data
        for j in range(471):
            x.append([])
            for t in range(18):
                for s in range(9):
                    x[471*i+j].append(data[t][480*i+j+s])
            y.append(data[9][480*i+j+9])
    return (np.array(x),np.array(y))

def load_testing_data(filename):
    row = genfromtxt(filename,delimiter=',')
    row = np.nan_to_num(row)
    data, n_row = [], 0
    for r in row:
        if n_row % 18 == 0:
            data.append([])
        for i in range(2,11):
           data[n_row//18].append(float(r[i]))
        n_row += 1
    data = np.array(data)
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

            if i % 1000 == 0:
                print ('iteration: %d | Cost: %f | Loss: %f ' % ( i,cost_a,loss_a))
        return w

if __name__=='__main__':
    x,y = load_training_data(sys.argv[1])
    test_data = load_testing_data(sys.argv[2])
    x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
    x = np.c_[x,x**2]
    
    # weight,learning rate
    w = np.zeros(len(x[0]))
    print(x.shape)
    #adjustment here
    k = Regression(0.1,120000).training(x,y,w)

    with open(sys.argv[3],"w+") as result:
        print("id,value",file=result)
        for i in range(len(test_data)):
            print("id_"+str(i)+","+str(np.dot(k,np.r_[test_data[i],test_data[i]**2])),file=result)

