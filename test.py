import numpy as np
import tensorflow as tf
import argparse
import time
import cv2

from util import tile_images, restore_constants
from train import build
from tensorflow.examples.tutorials.mnist import input_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    # get MNIST data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # restore configuration
    constants = restore_constants(args.config)

    # make network
    reconstruct, generate, train = build(constants)

    sess = tf.Session()
    sess.__enter__()

    # restore parameters
    saver = tf.train.Saver()
    saver.restore(sess, args.model)

    batch_images, _ = mnist.train.next_batch(32)
    batch_images = np.reshape(batch_images, [32] + constants.IMAGE_SIZE) 

    # reconstruction
    reconst, latent = reconstruct(batch_images)

    # show reconstructed images
    reconst_images = np.array(reconst * 255, dtype=np.uint8)
    reconst_tiled_images = tile_images(reconst_images)
    cv2.imshow('test', reconst_tiled_images)

    # show original images
    original_images = np.array(batch_images * 255, dtype=np.uint8)
    original_tiled_images = tile_images(original_images)
    cv2.imshow('original', original_tiled_images)

    while cv2.waitKey(10) < 10:
        time.sleep(0.1)

if __name__ == '__main__':
    main()
