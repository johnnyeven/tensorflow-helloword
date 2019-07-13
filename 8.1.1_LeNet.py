import tensorflow as tf
import numpy as np
import utils.input_data as input_data

mnist = input_data.read_data_sets("./mnist_data", one_hot=True)

batch_size = 100
learning_rate = 0.01
learning_rate_decay = 0.99
max_steps = 30000
model_save_path = './models/8.1.1/'
model_name = 'mnist_cnn'


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
        reshaped = tf.reshape(pool2, [tf.shape(pool2)[0], nodes])

    # layer 5
    with tf.variable_scope("F5-full", reuse=reuse):
        full1_weights = tf.get_variable("weight", [nodes, 512], initializer=tf.truncated_normal_initializer(stddev=0.1))
        full1_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))

        tf.add_to_collection("losses", regularizer(full1_weights))

        if avg_class is None:
            full1 = tf.nn.relu(tf.matmul(reshaped, full1_weights) + full1_biases)
        else:
            full1 = tf.nn.relu(tf.matmul(reshaped, avg_class.average(full1_weights)) + avg_class.average(full1_biases))

    # layer 6
    with tf.variable_scope("F6-full", reuse=reuse):
        full2_weights = tf.get_variable("weight", [512, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
        full2_biases = tf.get_variable("bias", [10], initializer=tf.constant_initializer(0.1))

        tf.add_to_collection("losses", regularizer(full2_weights))

        if avg_class is None:
            result = tf.matmul(full1, full2_weights) + full2_biases
        else:
            result = tf.matmul(full1, avg_class.average(full2_weights)) + avg_class.average(full2_biases)
    return result


with tf.variable_scope('placeholders'):
    x = tf.placeholder(tf.float32, [None, 28, 28, 1], name="input")
    y_ = tf.placeholder(tf.float32, [None, 10], name="label")

regularizer = tf.contrib.layers.l2_regularizer(0.0001)
y = hidden_layers(x, regularizer, avg_class=None, reuse=False)

# moving average
training_step = tf.Variable(0, trainable=False)
variable_averages = tf.train.ExponentialMovingAverage(0.99, training_step)
variables_averages_op = variable_averages.apply(tf.trainable_variables())
average_y = hidden_layers(x, regularizer, variable_averages, reuse=True)

# loss
with tf.variable_scope('loss'):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    loss = tf.reduce_mean(cross_entropy) + tf.add_n(tf.get_collection('losses'))

# train
learning_rate = tf.train.exponential_decay(learning_rate, training_step, mnist.train.num_examples / batch_size,
                                           learning_rate_decay, staircase=True)
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=training_step)
train_op = tf.group([train, variables_averages_op])

# prediction
with tf.variable_scope('prediction'):
    prediction_index = tf.argmax(y, 1, name='prediction_index')
    prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name='accuracy')

saver = tf.train.Saver()

# 最多占gpu资源的70%
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# 开始不会给tensorflow全部gpu资源 而是按需增加
gpu_options.allow_growth = True
config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1, max_steps + 1):
        if i % batch_size == 0:
            x_valid, y_valid = mnist.validation.next_batch(batch_size)
            reshaped_valid = np.reshape(x_valid, (batch_size, 28, 28, 1))
            validate_feed = {x: reshaped_valid, y_: y_valid}
            validate_accuracy = sess.run(accuracy, feed_dict=validate_feed)
            print("After %d training step(s), validation accuracy using average model is %g%%" %
                  (i, validate_accuracy * 100))

        x_train, y_train = mnist.train.next_batch(batch_size)
        reshaped_train = np.reshape(x_train, (batch_size, 28, 28, 1))
        train_feed = {x: reshaped_train, y_: y_train}
        sess.run(train_op, feed_dict=train_feed)

        if i % 1000 == 0:
            saver.save(sess, save_path=model_save_path + model_name, global_step=training_step)
