import tensorflow as tf


with tf.variable_scope('one'):
    a1 = tf.get_variable('a', [1], dtype=tf.float64, initializer=tf.constant_initializer(1.0))
    # error occurred because of the
    a2 = tf.get_variable('a', [1])
