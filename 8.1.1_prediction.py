import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils.input_data as input_data

model_path = './models/8.1.1/'
meta_graph_path = './models/8.1.1/mnist_cnn-30000.meta'
mnist = input_data.read_data_sets('./mnist_data', one_hot=True)


def hidden_layers(inputs, regularizer, avg_class, reuse):
    # layer 1
    with tf.variable_scope("C1-conv", reuse=reuse):
        conv1_weights = tf.get_variable("weight", [5, 5, 1, 32],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))

        conv1 = tf.nn.conv2d(inputs, conv1_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # layer 2
    with tf.name_scope("S2-max_pool"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # layer 3
    with tf.variable_scope("C3-conv", reuse=reuse):
        conv2_weights = tf.get_variable("weight", [5, 5, 32, 64],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # layer 4
    with tf.name_scope("S4-max_pool"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        shape = pool2.get_shape().as_list()  # shape = [100, 7, 7, 64]
        nodes = shape[1] * shape[2] * shape[3]  # nodes=3136
        reshaped = tf.reshape(pool2, [shape[0], nodes])

    # layer 5
    with tf.variable_scope("F5-full", reuse=reuse):
        full1_weights = tf.get_variable("weight", [nodes, 512], initializer=tf.truncated_normal_initializer(stddev=0.1))
        full1_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))

        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(full1_weights))

        if avg_class is None:
            full1 = tf.nn.relu(tf.matmul(reshaped, full1_weights) + full1_biases)
        else:
            full1 = tf.nn.relu(tf.matmul(reshaped, avg_class.average(full1_weights)) + avg_class.average(full1_biases))

    # layer 6
    with tf.variable_scope("F6-full", reuse=reuse):
        full2_weights = tf.get_variable("weight", [512, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
        full2_biases = tf.get_variable("bias", [10], initializer=tf.constant_initializer(0.1))

        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(full2_weights))

        if avg_class is None:
            result = tf.matmul(full1, full2_weights) + full2_biases
        else:
            result = tf.matmul(full1, avg_class.average(full2_weights)) + avg_class.average(full2_biases)
        print(result.name)
    return result


def visualization(image_array):
    plt.imshow(image_array, cmap="gray")
    plt.show()


x = tf.placeholder(tf.float32, [1, 28, 28, 1], name="input")
y_ = tf.placeholder(tf.float32, [1, 10], name="output")
y = hidden_layers(x, regularizer=None, avg_class=None, reuse=False)

# prediction
prediction_index = tf.argmax(y, 1)
prediction = tf.equal(prediction_index, tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
variable_averages = tf.train.ExponentialMovingAverage(.99)

graph = tf.train.Saver(variable_averages.variables_to_restore())

# 最多占gpu资源的70%
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# 开始不会给tensorflow全部gpu资源 而是按需增加
gpu_options.allow_growth = True
config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

with tf.Session(config=config) as sess:
    sess.graph.finalize()
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        graph.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise FileNotFoundError

    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    print("the latest model global step is %s" % global_step)

    rand = random.randint(1, 4999)
    mnist.validation.next_batch(rand)
    valid_x, valid_y = mnist.validation.next_batch(1)
    valid_x_img = np.reshape(valid_x * 255, (28, 28))
    visualization(valid_x_img)
    valid_x_reshaped = np.reshape(valid_x, (1, 28, 28, 1))
    valid_feed = {x: valid_x_reshaped, y_: valid_y}

    print(sess.run(prediction_index, feed_dict=valid_feed))
    # print(valid_y)
    # print(sess.run(accuracy, feed_dict=valid_feed))
