#!/bin/bash
wget https://www.csie.ntu.edu.tw/~b04902090/gensim_model
python train_preprocess.py --label $1
python gensim_train.py