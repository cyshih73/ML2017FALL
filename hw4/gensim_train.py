import random
import pickle
import argparse
import numpy as np
from gensim.models import Word2Vec

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM, GRU, Bidirectional, Conv1D, AveragePooling1D, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import *

def read_embedding(original_sequences, word2idx, idx2vec, maxlen, dim):
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

def main():
    maxlen = 40
    data_labeled = pickle.load(open("data_labeled.dat", "rb"))
    temp_label = pickle.load(open("data_labeled_labels.dat", "rb"))

    sequence, label = data_labeled, temp_label

    #Gensim embeddings_matrix
    Gensim_model = Word2Vec.load('gensim_model')
    word2idx = {"_PAD": 0}
    vocab_list = [(k, Gensim_model.wv[k]) for k, v in Gensim_model.wv.vocab.items()]
    embeddings_matrix = np.zeros((len(Gensim_model.wv.vocab.items()) + 1, Gensim_model.vector_size))
    for i in range(len(vocab_list)):
        word = vocab_list[i][0]
        word2idx[word] = i + 1
        embeddings_matrix[i + 1] = vocab_list[i][1]

    sequences = read_embedding(sequence, word2idx, embeddings_matrix, maxlen, Gensim_model.vector_size)
    print(sequences.shape)

    #Model
    model = Sequential()
    model.add(Embedding(len(embeddings_matrix), Gensim_model.vector_size, weights=[embeddings_matrix],
        input_length=maxlen, trainable=True))
    model.add(Bidirectional(LSTM(256, activation='tanh', recurrent_dropout=0.4, dropout=0.4,
        return_sequences=True)))
    model.add(Bidirectional(LSTM(128, activation='tanh', recurrent_dropout=0.4, dropout=0.4,
        return_sequences=False)))

    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())

    callbacks = []
    callbacks.append(ModelCheckpoint('hw4_model-{epoch:02d}.h5', monitor='acc', verbose=1,
        save_best_only=True, mode='max'))
    model.fit(sequences, label, epochs=6, batch_size=256, callbacks=callbacks)

if __name__ == '__main__':
    main()
