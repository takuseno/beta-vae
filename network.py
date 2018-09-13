import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers import xavier_initializer as w_init
from tensorflow.contrib.layers import xavier_initializer_conv2d as conv_w_init


def _make_encoder(convs, fcs, activation, inputs, keep_prob, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse):
        out = inputs
        # convolution layers
        for _, ch, kernel, stride in convs:
            out = tf.layers.conv2d(out, ch, kernel_size=kernel, strides=stride,
                                   activation=activation, padding='same',
                                   kernel_initializer=conv_w_init())
            out = tf.nn.dropout(out, keep_prob)

        # retain dimension of the last cnn layer
        feature = out
        feature_size = out.shape[1] * out.shape[2] * out.shape[3]
        out = tf.reshape(out, [-1, feature_size])

        # fully connected layers
        for i, fc in enumerate(fcs):
            out = tf.layers.dense(out, fc, activation=activation,
                                  kernel_initializer=w_init())
            out = tf.nn.dropout(out, keep_prob)
    return out, feature.shape[1:]

def _make_decoder(convs, fcs, activation, latent,
                  feature_shape, keep_prob, reuse=False):
    with tf.variable_scope('decoder', reuse=reuse):
        out = latent
        # fully connected layers
        for fc in reversed(fcs):
            out = tf.layers.dense(out, fc, activation=activation,
                                  kernel_initializer=w_init())
            out = tf.nn.dropout(out, keep_prob)

        # enlarge dimension back to the last cnn layer
        feature_size = feature_shape[0] * feature_shape[1] * feature_shape[2]
        out = tf.layers.dense(out, feature_size, activation=activation,
                              kernel_initializer=w_init())
        out = tf.nn.dropout(out, keep_prob)
        out = tf.reshape(out, [-1] + list(feature_shape))

        # deconvolution layers
        for i, (ch, _, kernel, stride) in enumerate(reversed(convs)):
            out = tf.layers.conv2d_transpose(out, ch, kernel_size=kernel,
                                             strides=stride, padding='same',
                                             kernel_initializer=conv_w_init())
            if i != len(convs) - 1:
                out = activation(out)
                out = tf.nn.dropout(out, keep_prob)
    return out

def _make_latent(latent_size, inputs, deterministic, reuse=False):
    with tf.variable_scope('latent', reuse=reuse):
        # mean and standard deviation
        mu = tf.layers.dense(inputs, latent_size, kernel_initializer=w_init())
        log_std = tf.layers.dense(inputs, latent_size, kernel_initializer=w_init())
        # reparametization trick
        eps = tf.random_normal(tf.shape(log_std), 0.0, 1.0, dtype=tf.float32)
        latent = mu + eps * tf.sqrt(tf.exp(log_std)) * (1.0 - deterministic)
    return latent, mu, log_std

def make_encoder(convs, fcs, activation):
    return lambda *args, **kwargs: _make_encoder(convs, fcs, activation,\
                                                 *args, **kwargs)

def make_decoder(convs, fcs, activation):
    return lambda *args, **kwargs: _make_decoder(convs, fcs, activation,\
                                                 *args, **kwargs)

def make_latent(latent_size):
    return lambda *args, **kwargs: _make_latent(latent_size, *args, **kwargs)
