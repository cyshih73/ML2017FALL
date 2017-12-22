#!/bin/bash
wget https://www.csie.ntu.edu.tw/~b04902090/hw5_model.h5
python predict.py --test $1 --output $2 --model hw5_model.h5 --user user2id.npy --movie movie2id.npy
