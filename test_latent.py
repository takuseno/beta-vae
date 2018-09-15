import tensorflow as tf
import numpy as np
import argparse
import cv2
import time

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
    reconstruct, generate_from_latent, train = build(constants)

    sess = tf.Session()
    sess.__enter__()

    # restore parameters
    saver = tf.train.Saver()
    saver.restore(sess, args.model)

    batch_images, _ = mnist.train.next_batch(1)
    batch_images = np.reshape(batch_images, [1] + constants.IMAGE_SIZE)

    # reconstruction
    reconst, latent = reconstruct(batch_images)
    latent_range = np.linspace(-3.0, 3.0, num=20)

    image_rows = []
    for i in range(constants.LATENT_SIZE):
        # change specific element of latent variable
        tiled_latent = np.tile(latent[0].copy(), (20, 1))
        tiled_latent[:,i] = latent_range

        # reconstruct from latent variable
        reconst = generate_from_latent(tiled_latent)

        # tiling reconstructed images
        reconst_images = np.array(reconst * 255, dtype=np.uint8)
        reconst_tiled_images = tile_images(reconst_images, row=1)
        image_rows.append(reconst_tiled_images)

    # show reconstructed images
    image_rows = tile_images(np.array(image_rows), row=constants.LATENT_SIZE)
    cv2.imshow('test'.format(i), image_rows)

    while cv2.waitKey(10) < 10:
        time.sleep(0.1)

if __name__ == '__main__':
    main()
