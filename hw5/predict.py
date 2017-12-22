import argparse
import numpy as np
import pandas as pd
import keras.backend as K
from keras.models import load_model
from keras.engine.topology import Layer

def parse_args():
    parser = argparse.ArgumentParser('Matrix Factorization.')
    parser.add_argument('--test', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--user', required=True)
    parser.add_argument('--movie', required=True)
    return parser.parse_args()

class computing(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(computing, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask, axis=-1)
            s = K.sum(mask, axis=1)
            if K.equal(s, K.zeros_like(s)) is None: return K.mean(x, axis=1)
            else: return K.cast(K.sum(x * mask, axis=1) / K.sqrt(s), K.floatx())
        else: return K.mean(x, axis=1)

    def compute_mask(self, x, mask=None): return None
    def compute_output_shape(self, input_shape): return (input_shape[0], input_shape[-1])

    def config(self):
        base = super(computing, self).config()
        return dict(list(base.items()))

def rmse(y_true, y_pred):
    y_pred = K.clip(y_pred, 1.0, 5.0)
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def read_data(filename, user2id, movie2id):
    df = pd.read_csv(filename)
    df['UserID'] = df['UserID'].apply(lambda x: user2id[x])
    df['MovieID'] = df['MovieID'].apply(lambda x: movie2id[x])
    return df['TestDataID'], df[['UserID', 'MovieID']].values

def main(args):
    user2id = np.load(args.user)[()]
    movie2id = np.load(args.movie)[()]
    id, text_x = read_data(args.test, user2id, movie2id)

    model = load_model(args.model, custom_objects={'rmse': rmse, 'computing': computing})
    pred = model.predict([text_x[:, 0], text_x[:, 1]]).squeeze()
    pred = pred.clip(1.0, 5.0)

    df = pd.DataFrame({'TestDataID': id, 'Rating': pred}, columns=('TestDataID', 'Rating'))
    df.to_csv(args.output, index=False, columns=('TestDataID', 'Rating'))

if __name__ == '__main__':
    args = parse_args()
    main(args)
