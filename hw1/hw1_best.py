# coding: utf-8
import numpy as np
from numpy import *
import math
import random
import sys
import csv
feature = [7,8,9,10,14,15,16,17]
feature_squ = [8,9]

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
    	for s in range(9):
        	data = np.c_[data,data[:,9*feature.index(t)+s]**2]
    return np.concatenate((np.ones((data.shape[0],1)),data), axis=1)

if __name__=='__main__':
    test_data = load_testing_data(sys.argv[1])
    k = np.load("./weight.npy")

    with open(sys.argv[2],"w+") as result:
        print("id,value",file=result)
        for i in range(len(test_data)):
            print("id_"+str(i)+","+str(np.dot(k,test_data[i])),file=result)