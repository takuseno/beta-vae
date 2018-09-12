import numpy as np
import tensorflow as tf


def _make_encoder(convs, fcs, inputs, keep_prob, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse):
        out = inputs
        for _, ch, kernel, stride in convs:
            out = tf.layers.conv2d(out, ch, kernel_size=kernel, strides=stride,
                                   activation=tf.nn.relu, padding='same')
            out = tf.nn.dropout(out, keep_prob)
        feature = out
        feature_size = out.shape[1] * out.shape[2] * out.shape[3]
        out = tf.reshape(out, [-1, feature_size])
        for i, fc in enumerate(fcs):
            out = tf.layers.dense(out, fc, activation=tf.nn.relu)
            out = tf.nn.dropout(out, keep_prob)
    return out, feature.shape[1:]

def _make_decoder(convs, fcs, latent, feature_shape, keep_prob, reuse=False):
    with tf.variable_scope('decoder', reuse=reuse):
        out = latent
        for fc in reversed(fcs):
            out = tf.layers.dense(out, fc, activation=tf.nn.relu)
            out = tf.nn.dropout(out, keep_prob)

        feature_size = feature_shape[0] * feature_shape[1] * feature_shape[2]
        out = tf.layers.dense(out, feature_size, activation=tf.nn.relu)
        out = tf.nn.dropout(out, keep_prob)
        out = tf.reshape(out, [-1] + list(feature_shape))

        for i, (ch, _, kernel, stride) in enumerate(reversed(convs)):
            out = tf.layers.conv2d_transpose(
                out, ch, kernel_size=kernel, strides=stride, padding='same')
            if i != len(convs) - 1:
                out = tf.nn.relu(out)
                out = tf.nn.dropout(out, keep_prob)
    return out

def _make_latent(latent_size, inputs, reuse=False):
    with tf.variable_scope('latent', reuse=reuse):
        # latent
        mu = tf.layers.dense(inputs, latent_size)
        log_std = tf.layers.dense(inputs, latent_size)
        # reparametization trick
        eps = tf.random_normal(tf.shape(log_std))
        latent = mu + eps * tf.sqrt(tf.exp(log_std))
    return latent, mu, log_std

def make_encoder(convs, fcs):
    return lambda *args, **kwargs: _make_encoder(convs, fcs, *args, **kwargs)

def make_decoder(convs, fcs):
    return lambda *args, **kwargs: _make_decoder(convs, fcs, *args, **kwargs)

def make_latent(latent_size):
    return lambda *args, **kwargs: _make_latent(latent_size, *args, **kwargs)
