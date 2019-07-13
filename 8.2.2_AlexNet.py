import tensorflow as tf
import math
import time
from datetime import datetime

batch_size = 32
num_batches = 100


def hidden_layer(images):
    parameters = []

    # C1 layer
    with tf.name_scope("conv1"):
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype=tf.float32, stddev=0.1), name="weights")
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32), name="biases")
        conv1_result = tf.nn.relu(tf.nn.bias_add(conv, biases))

        print("C1 layer conv result shape = %g" % conv1_result.get_shape().as_list())
        parameters += [kernel, biases]

        lrn1 = tf.nn.lrn(conv1_result, 4, alpha=0.001 / 9.0, beta=0.75, name="lrn")
        pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool")

        print("C1 layer pool shape = %g" % pool1.get_shape().as_list())

    # C2 layer
    with tf.name_scope("conv2"):
        kernel = tf.Variable(tf.truncated_normal([5, 5, 96, 256], dtype=tf.float32, stddev=0.1), name="weights")
        conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), name="biases")
        conv2_result = tf.nn.relu(tf.nn.bias_add(conv, biases))

        print("C2 layer conv result shape = %g" % conv2_result.get_shape().as_list())
        parameters += [kernel, biases]

        lrn2 = tf.nn.lrn(conv2_result, 4, alpha=0.001 / 9.0, beta=0.75, name="lrn")
        pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool")

        print("C2 layer pool shape = %g" % pool2.get_shape().as_list())

    # C3 layer
    with tf.name_scope("conv3"):
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 384], dtype=tf.float32, stddev=0.1), name="weights")
        conv = tf.nn.conv2d(pool2, kernel, strides=[1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), name="biases")
        conv3_result = tf.nn.relu(tf.nn.bias_add(conv, biases))

        print("C3 layer conv result shape = %g" % conv3_result.get_shape().as_list())
        parameters += [kernel, biases]

    # C4 layer
    with tf.name_scope("conv4"):
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 384], dtype=tf.float32, stddev=0.1), name="weights")
        conv = tf.nn.conv2d(conv3_result, kernel, strides=[1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), name="biases")
        conv4_result = tf.nn.relu(tf.nn.bias_add(conv, biases))

        print("C4 layer conv result shape = %g" % conv4_result.get_shape().as_list())
        parameters += [kernel, biases]

    # C5 layer
    with tf.name_scope("conv5"):
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=0.1), name="weights")
        conv = tf.nn.conv2d(conv4_result, kernel, strides=[1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), name="biases")
        conv5_result = tf.nn.relu(tf.nn.bias_add(conv, biases))

        print("C5 layer conv result shape = %g" % conv5_result.get_shape().as_list())
        parameters += [kernel, biases]

        pool5 = tf.nn.max_pool(conv5_result, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool")

        print("C5 layer pool shape = %g" % pool5.get_shape().as_list())

        pool_shape = pool5.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool5, [tf.shape(pool5)[0], nodes])

    # FC6 layer
    with tf.name_scope("fc6"):
        weights = tf.Variable(tf.truncated_normal([nodes, 4096], dtype=tf.float32, stddev=0.1), name="weights")
        biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), name="biases")
        fc6_result = tf.nn.relu(tf.matmul(reshaped, weights) + biases)

        print("FC6 layer result shape = %g" % fc6_result.get_shape().as_list())
        parameters += [weights, biases]

    # FC7 layer
    with tf.name_scope("fc7"):
        weights = tf.Variable(tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=0.1), name="weights")
        biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), name="biases")
        fc7_result = tf.nn.relu(tf.matmul(fc6_result, weights) + biases)

        print("FC7 layer result shape = %g" % fc7_result.get_shape().as_list())
        parameters += [weights, biases]

    return fc7_result, parameters
