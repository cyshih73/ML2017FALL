import numpy as np
import sys
import random
import os
os.environ["THEANO_FLAGS"] = "device=gpu0"

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
import keras.preprocessing.image as img
np.random.seed(7)

def load_data(filename):
    print("------reading data------")
    x, y = [], []
    with open(filename) as fp:
        fp.readline()
        for line in fp:
            cols = line.split(',')
            y.append(int(cols[0]))
            x.append(list(map(int, cols[1].split())))
    return (np.array(x), np.array(y))

def main(args):
    train_x,train_y = load_data(sys.argv[1])
    print("data shape = ",end='')
    print(train_x.shape,train_y.shape)

    # reshape and normalize
    train_x = (train_x.reshape(train_x.shape[0], 48, 48, 1).astype('float32'))/255
    train_y = np_utils.to_categorical(train_y)

    print("reshape to : ",end='')
    print(train_x.shape,train_y.shape)

    model = Sequential()
    #1st layer
    model.add(Conv2D(filters=50, kernel_size=(5, 5), input_shape=(48, 48, 1), activation='relu')) 
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    #2nd layer
    model.add(Conv2D(filters=150, kernel_size=(3, 3), activation='relu')) 
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    #3rd layer
    model.add(Conv2D(filters=500, kernel_size=(3, 3), activation='relu'))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    #4th layer
    model.add(Conv2D(filters=500, kernel_size=(3, 3), activation='relu'))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(518, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(259, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))    
    model.add(Dense(7))    
    model.add(Activation('softmax'))  
    print("model summary :")  
    model.summary()    

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
    imggen = img.ImageDataGenerator(        
        rotation_range = 10,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=[0.8, 1.2],
        shear_range=0.2)
    imggen.fit(train_x)

    print("--------training--------")
    model.fit_generator(imggen.flow(train_x, train_y, batch_size=128),
        steps_per_epoch=len(train_x)/16,epochs=120)
    model.save('hw3_model.h5')

if __name__ == '__main__':
    main(sys.argv)
