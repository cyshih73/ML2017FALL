#!/bin/bash
wget https://www.csie.ntu.edu.tw/~b04902090/hw3_model.h5
python3 predict.py $1 $2 hw3_model.h5
