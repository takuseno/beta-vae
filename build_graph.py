import tensorflow as tf


def build_graph(encoder,
                decoder,
                sample_latent,
                image_size,
                latent_size,
                lr,
                scope='vae'):
    with tf.variable_scope(scope):
        input_ph = tf.placeholder(tf.float32, [None] + image_size, name='input')
        beta_ph = tf.placeholder(tf.float32, [], name='beta')
        latent_ph = tf.placeholder(tf.float32, [None, latent_size], name='latent')
        keep_prob_ph = tf.placeholder(tf.float32, [], name='keep_prob')
        deterministic_ph = tf.placeholder(tf.float32, [], name='deterministic')

        # network processes inputs
        encoded, feature_shape = encoder(input_ph, keep_prob_ph)
        latent, mu, log_std = sample_latent(encoded, deterministic_ph)
        reconst_logits = decoder(latent, feature_shape, keep_prob_ph)

        # reconstruction image
        reconst = tf.nn.sigmoid(reconst_logits)

        # from latent to reconstruction
        generate_logits = decoder(latent_ph, feature_shape, keep_prob_ph, reuse=True)
        generate = tf.nn.sigmoid(generate_logits)

        # parameters
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

        # reconstruction loss
        batch_size = tf.shape(input_ph)[0]
        flatten_input = tf.reshape(input_ph, [batch_size, -1])
        flatten_reconst = tf.reshape(reconst_logits, [batch_size, -1])
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=flatten_input,
                                                          logits=flatten_reconst)
        reconst_loss = tf.reduce_mean(tf.reduce_sum(entropy, axis=1))
        # kl divergence
        kl = 0.5 * (-log_std + tf.square(mu) + tf.exp(log_std) - 1)
        kl_penalty = tf.reduce_mean(tf.reduce_sum(kl, axis=1))
        # loss
        loss = reconst_loss + beta_ph * kl_penalty

        opt = tf.train.AdamOptimizer(lr)
        opt_expr = opt.minimize(loss, var_list=var_list)

    def reconstruct(inputs):
        feed_dict = {
            input_ph: inputs,
            keep_prob_ph: 1.0,
            deterministic_ph: 1.0
        }
        sess = tf.get_default_session()
        return sess.run([reconst, mu], feed_dict)

    def generate_from_latent(latent):
        feed_dict = {
            latent_ph: latent,
            keep_prob_ph: 1.0,
            deterministic_ph: 1.0
        }
        sess = tf.get_default_session()
        return sess.run(generate, feed_dict=feed_dict)

    def train(inputs, keep_prob=0.5, beta=1.0):
        feed_dict = {
            input_ph: inputs,
            beta_ph: beta,
            keep_prob_ph: keep_prob,
            deterministic_ph: 0.0
        }
        sess = tf.get_default_session()
        return sess.run([loss, opt_expr], feed_dict=feed_dict)[0]

    return reconstruct, generate_from_latent, train
