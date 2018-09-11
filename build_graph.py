import tensorflow as tf

from network import encoder, decoder


def build_graph(network, latent_size=20, scope='vae'):
    with tf.variable_scope(scope):
        input_ph = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input')
        beta_ph = tf.placeholder(tf.float32, [], name='beta')
        latent_ph = tf.placeholder(tf.float32, [None, latent_size], name='latent')

        # network processes inputs
        feature,\
        latent,\
        reconst_logits,\
        mu,\
        log_std = network(input_ph, latent_size)

        # reconstruction image
        reconst = tf.nn.sigmoid(reconst_logits)

        # from latent to reconstruction
        from_latent = decoder(latent_ph, reuse=True)

        # parameters
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

        # reconstruction loss
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=input_ph,
                                                          logits=reconst_logits)
        reconst_loss = tf.reduce_mean(tf.reduce_mean(entropy, axis=1))
        # kl divergence
        kl = 0.5 * (-log_std + tf.square(mu) + tf.exp(log_std) - 1)
        kl_penalty = tf.reduce_mean(tf.reduce_mean(kl, axis=1))
        # loss
        loss = reconst_loss + beta_ph * kl_penalty

        opt = tf.train.AdamOptimizer(5e-4)
        opt_expr = opt.minimize(loss, var_list=var_list)

    def reconstruct(inputs):
        feed_dict = {
            input_ph: inputs
        }
        sess = tf.get_default_session()
        return sess.run([reconst, latent], feed_dict)

    def reconstruct_from_latent(latent):
        feed_dict = {
            latent_ph: latent
        }
        sess = tf.get_default_session()
        return sess.run(from_latent, feed_dict=feed_dict)

    def train(inputs, beta=1.0):
        feed_dict = {
            input_ph: inputs,
            beta_ph: beta
        }
        sess = tf.get_default_session()
        return sess.run([loss, opt_expr], feed_dict=feed_dict)[0]

    return reconstruct, reconstruct_from_latent, train
