# coding: utf-8
import numpy as np
import sys, glob, argparse
from skimage import transform, data, io

def parse_args():
    parser = argparse.ArgumentParser('HW6 - PCA of colored faces.')
    parser.add_argument('--imgs', required=True)
    parser.add_argument('--eig', type=int, default=4)
    parser.add_argument('--target', required=True)
    return parser.parse_args()

def convert255(array):
    array -= np.min(array)
    array /= np.max(array)
    array = (array * 255).astype(np.uint8)
    return array

def main(args):
    images = []
    for fn in glob.glob(args.imgs + '/*.jpg'):
        image = io.imread(fn)
        images.append(image.flatten())
    images = np.array(images)
    print("Done Reading. Image shape = ", images.shape)
    images_mean = np.mean(images, axis=0)

    print('SVDing')
    U, s, V = np.linalg.svd((images - images_mean).T, full_matrices=False)

    #reconstructing
    target = io.imread(args.imgs+'/'+args.target).flatten() - images_mean
    result = np.zeros(target.shape[0])
    for i in range(args.eig): result += (np.dot(target, U[:, i]) * U[:, i]).T

    result = convert255(result + images_mean).reshape(600,600,3)
    io.imsave("./reconstruction.jpg", result)

if __name__ == '__main__':
    args = parse_args()
    main(args)
