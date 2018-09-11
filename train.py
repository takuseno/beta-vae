import numpy as np
import tensorflow as tf
import cv2
import constants

from build_graph import build_graph
from network import make_encoder, make_decoder, make_latent
from tensorflow.examples.tutorials.mnist import input_data


def tile_images(images, row=2):
    shape = images.shape[1:]
    column = int(images.shape[0] / row)
    height = shape[0]
    width = shape[1]
    tile_height = row * height
    tile_width = column * width
    output = np.zeros((tile_height, tile_width, shape[-1]), dtype=np.uint8)
    for i in range(row):
        for j in range(column):
            image = images[i*column+j]
            output[i*height:(i+1)*height,j*width:(j+1)*width] = image
    return output

def main():
    # get MNIST data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # make networks
    encoder = make_encoder(constants.CONVS, constants.FC)
    decoder = make_decoder(constants.CONVS, constants.FC)
    sample_latent = make_latent(constants.LATENT_SIZE)

    # build graphs
    reconstruct,\
    reconstruct_from_latent,\
    train = build_graph(
        encoder=encoder,
        decoder=decoder,
        sample_latent=sample_latent,
        image_size=constants.IMAGE_SIZE,
        latent_size=constants.LATENT_SIZE,
        lr=constants.LR
    )

    sess = tf.Session()
    sess.__enter__()
    sess.run(tf.global_variables_initializer())

    # constant variables
    batch_size = constants.BATCH_SIZE
    image_size = constants.IMAGE_SIZE

    # start training
    for i in range(10000):
        batch_images, _ = mnist.train.next_batch(batch_size)
        batch_images = np.reshape(batch_images, [batch_size] + image_size) 
        loss = train(batch_images, beta=constants.BETA)
        print(loss)
        if i % 100 == 0:
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

            if cv2.waitKey(10) > 0:
                pass

if __name__ == '__main__':
    main()
