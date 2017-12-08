import pickle
import logging
import argparse
import itertools
import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

def load_labeled(filename):
    x, y = [], []
    with open(filename) as fp:
        for line in fp:
            cols = line.split(' +++$+++ ')
            y.append(int(cols[0]))
            cols[1] = cols[1].replace('?•','').encode('ascii',errors = 'ignore')
            cols[1] = text_to_word_sequence(str(cols[1]),
                filters='1234567890\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
            words = []
            for word in cols[1][1:-1]:
                word_groupby = ''.join(ch for ch, _ in itertools.groupby(word))
                words.append(word_groupby)
            x.append(words)
    print("done reading")
    return (x, y)

def parse_args():
    parser = argparse.ArgumentParser(description='Twitter text sentiment data preprocess.')
    parser.add_argument('--label')
    return parser.parse_args()

#python data_preprocess.py --labeled labeled.txt --unlabeled nolabeled.txt --test data_test.txt

def main(args):
    data_labeled, label = load_labeled(args.label)
    #data_unlabeled = load_unlabeled(args.unlabel)
    pickle.dump(data_labeled, open("data_labeled.dat", "wb"))
    pickle.dump(label, open("data_labeled_labels.dat", "wb"))

if __name__ == '__main__':
    args = parse_args()
    main(args)
