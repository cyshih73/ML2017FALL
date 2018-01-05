import argparse
import numpy as np
from sklearn.cluster import KMeans
from keras.models import Model, load_model

def parse_args():
    parser = argparse.ArgumentParser('Image Clustering.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--out', required=True)
    return parser.parse_args()

def main(args):
    images = np.load(args.data) / 255.
    print("image shape = ", images.shape)

    encoder = load_model("./encoder_model.h5")
    encoded_imgs = encoder.predict(images)
    print("done predict: ", encoded_imgs.shape)

    tags = KMeans(n_clusters=2).fit_predict(encoded_imgs)

    zero, one = 0, 0
    row = np.genfromtxt(args.test, delimiter=',', dtype=int, skip_header=1)
    print("printing answers")
    with open(args.out,"w+") as output:
        print("ID,Ans",file=output)
        for i in range(row.shape[0]):
            if tags[row[i][1]] == tags[row[i][2]]:
                print(str(row[i][0])+","+str(1), file=output)
                one += 1
            else:
                print(str(row[i][0])+","+str(0), file=output)
                zero += 1
    print("zero", zero, "one", one)

if __name__ == '__main__':
    args = parse_args()
    main(args)
