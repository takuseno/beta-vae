import numpy as np
import tensorflow as tf


def encoder(inputs, reuse=False):
    w_init = tf.orthogonal_initializer(np.sqrt(2.0))
    with tf.variable_scope('encoder', reuse=reuse):
        out = tf.layers.conv2d(inputs, 32, kernel_size=(4, 4), strides=2,
                               activation=tf.nn.relu, padding='valid',
                               kernel_initializer=w_init)
        out = tf.reshape(out, [-1, 13*13*32])
        feature = tf.layers.dense(
            out, 256, activation=tf.nn.relu, kernel_initializer=w_init)
    return feature

def decoder(latent, reuse=False):
    w_init = tf.orthogonal_initializer(np.sqrt(2.0))
    with tf.variable_scope('decoder', reuse=reuse):
        out = tf.layers.dense(
            latent, 256, kernel_initializer=w_init, activation=tf.nn.relu)
        out = tf.layers.dense(
            latent, 13*13*32, kernel_initializer=w_init, activation=tf.nn.relu)
        out = tf.reshape(out, [-1, 13, 13, 32])
        out = tf.layers.conv2d_transpose(
            out, 1, kernel_size=(4, 4), strides=2, padding='valid',
            kernel_initializer=w_init)
    return out

def _make_network(inputs, latent_size):
    w_init = tf.orthogonal_initializer(np.sqrt(2.0))

    # encoding
    feature = encoder(inputs)

    # latent
    mu = tf.layers.dense(feature, latent_size, kernel_initializer=w_init)
    log_std = tf.layers.dense(feature, latent_size, kernel_initializer=w_init)
    # reparametization trick
    eps = tf.random_normal(tf.shape(log_std))
    latent = mu + eps * tf.sqrt(tf.exp(log_std))

    # decoding
    reconst = decoder(latent)

    return feature, latent, reconst, mu, log_std

def make_network():
    return lambda inputs, latent_size: _make_network(inputs, latent_size)
