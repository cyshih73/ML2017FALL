#!/bin/bash
wget https://www.csie.ntu.edu.tw/~b04902090/hw4_model-06.h5
wget https://www.csie.ntu.edu.tw/~b04902090/gensim_model
python test_preprocess.py --test $1
python predict.py $2