import tensorflow as tf
import numpy as np

# filter_weight = tf.get_variable("weights", [2, 2, 1, 1], initializer=tf.constant_initializer([[-1, 4], [2, 1]]))
filter_weight = tf.Variable(tf.constant([[-1, 4], [2, 1]], dtype=tf.float32, shape=[2, 2, 1, 1]), name="weights")
# biases = tf.get_variable("bias", [1], initializer=tf.constant_initializer(1))
biases = tf.Variable(tf.constant(1, dtype=tf.float32, shape=[1]), name="bias")

x = tf.placeholder('float32', [1, None, None, 1])
M = np.array(
    [
        [
            [2], [1], [2], [-1]
        ],
        [
            [0], [-1], [3], [0]
        ],
        [
            [2], [1], [-1], [4]
        ],
        [
            [-2], [0], [-3], [4]
        ]
    ], dtype="float32").reshape([1, 4, 4, 1])

conv = tf.nn.conv2d(x, filter_weight, strides=[1, 1, 1, 1], padding="SAME")
add_bias = tf.nn.bias_add(conv, biases)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    M_conv = sess.run(add_bias, feed_dict={x: M})

    print("M after convolution: \n", M_conv)
