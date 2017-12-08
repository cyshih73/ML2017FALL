import pickle
import logging
import argparse
import itertools
import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

def load_data(filename, Test=False):
    x = []
    with open(filename) as fp:
        if Test is True:
            fp.readline()
        for line in fp:
            cols = line.split(',', 1)[1]
            cols = cols.replace('?•','').encode('ascii',errors = 'ignore')
            cols = text_to_word_sequence(str(cols),
                filters='1234567890\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
            words = []
            for word in cols[1:-1]:
                word_groupby = ''.join(ch for ch, _ in itertools.groupby(word))
                words.append(word_groupby)
            x.append(words)
    print("done reading")
    return (x)

def parse_args():
    parser = argparse.ArgumentParser(description='Twitter text sentiment data preprocess.')
    parser.add_argument('--test')
    return parser.parse_args()

#python data_preprocess.py --labeled labeled.txt --unlabeled nolabeled.txt --test data_test.txt

def main(args):
    data_test = load_data(args.test, Test=True)
    pickle.dump(data_test, open("data_test.dat", "wb"))

if __name__ == '__main__':
    args = parse_args()
    main(args)
