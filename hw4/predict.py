import pickle
import sys
import numpy as np
from gensim.models import Word2Vec

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM, GRU, Bidirectional, Conv1D, AveragePooling1D, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import *
from keras.models import load_model

np.random.seed(7)

def read_embedding(original_sequences, word2idx, idx2vec, maxlen):
    sequences = []
    for words in original_sequences:
        count = 0
        sequence = []
        for word in words:
            if count < maxlen and word in word2idx:
                sequence.append(int(word2idx[word]))
                count = count + 1
        while count < maxlen:
            sequence.append(int(0))
            count = count + 1
        sequences.append(sequence)
    return np.array(sequences)

def main(args):
    maxlen = 40
    data_test = pickle.load(open("data_test.dat", "rb"))

    #Gensim embeddings_matrix
    Gensim_model = Word2Vec.load('gensim_model')
    word2idx = {"_PAD": 0}
    vocab_list = [(k, Gensim_model.wv[k]) for k, v in Gensim_model.wv.vocab.items()]
    embeddings_matrix = np.zeros((len(Gensim_model.wv.vocab.items()) + 1, Gensim_model.vector_size))
    for i in range(len(vocab_list)):
        word = vocab_list[i][0]
        word2idx[word] = i + 1
        embeddings_matrix[i + 1] = vocab_list[i][1]

    sequences_test = read_embedding(data_test, word2idx, embeddings_matrix, maxlen)

    model = load_model('hw4_model-06.h5')
    print("predicting")
    prediction = model.predict(sequences_test, batch_size=512)
    print(prediction.shape)
    with open(sys.argv[1],"w+") as result:
        print("id,label",file=result)
        for i in range(len(prediction)):
            if prediction[i] > float(0.5):
                print(str(i)+","+str(int(1)),file=result)
            else:
                print(str(i)+","+str(int(0)),file=result)

if __name__ == '__main__':
    main(sys.argv)
