import tensorflow as tf
import numpy as np
import time
import math
import utils.cifar10_data as cifar10_data

max_steps = 4000
batch_size = 100
num_examples_for_eval = 10000
data_dir = "./cifar10_data"


def variable_with_weight_loss(shape, stddev, wl):
    weight = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(weight), wl, name="weight_loss")
        tf.add_to_collection("losses", weight_loss)
    return weight


images_train, labels_train = cifar10_data.inputs(data_dir, batch_size, distorted=True)
images_test, labels_test = cifar10_data.inputs(data_dir, batch_size, distorted=None)

x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
y_ = tf.placeholder(tf.int32, [batch_size])

# CNN layer 1
kernel1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding="SAME")
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

# CNN layer 2
kernel2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding="SAME")
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value

# MLP layer 1
weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc_1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)

# MLP layer 2
weight2 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(fc_1, weight2) + fc_bias2)

# MLP layer 3
weight3 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, wl=0.0)
fc_bias3 = tf.Variable(tf.constant(0.0, shape=[10]))
y = tf.add(tf.matmul(local4, weight3), fc_bias3)

# loss
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.cast(y_, tf.int64))
weights_with_l2_loss = tf.add_n(tf.get_collection("losses"))
loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss

# train
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

# top_k
top_k_op = tf.nn.in_top_k(y, y_, 1)

# run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    tf.train.start_queue_runners()

    for step in range(max_steps):
        start_time = time.time()

        image_batch, label_batch = sess.run([images_train, labels_train])
        _, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch, y_: label_batch})

        end_time = time.time()
        duration = end_time - start_time

        if step % batch_size == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)

            print("step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)" % (
                step, loss_value, examples_per_sec, sec_per_batch))

    num_batch = int(math.ceil(num_examples_for_eval / batch_size))
    true_count = 0
    total_sample_count = step * batch_size

    for j in range(num_batch):
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions = sess.run([top_k_op], feed_dict={x: image_batch,
                                                      y_: label_batch})
        true_count += np.sum(predictions)

    # 打印正确率信息
    print("accuracy = %.3f%%" % ((true_count / total_sample_count) * 100))
