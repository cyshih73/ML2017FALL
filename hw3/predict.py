# coding: utf-8
import numpy as np
from numpy import *
import sys
from keras.models import load_model
def load_data(filename):
    print("------reading data------")
    x = []
    with open(filename) as fp:
        fp.readline()
        for line in fp:
            cols = line.split(',')
            x.append(list(map(int, cols[1].split())))
    return (np.array(x))

def main(args):
    #python predict.py test.csv result.csv [modelname]
    test_x = load_data(sys.argv[1])
    #reshape
    test_x = (test_x.reshape(test_x.shape[0], 48, 48, 1).astype('float32'))/255
    model = load_model(sys.argv[3])
    prediction = model.predict(test_x)

    print(prediction.shape)
    with open(sys.argv[2],"w+") as result:
        print("id,label",file=result)
        for i in range(test_x.shape[0]):
            j = prediction[i].argmax()
            print(str(i)+","+str(j),file=result)

if __name__=='__main__':
    main(sys.argv)
