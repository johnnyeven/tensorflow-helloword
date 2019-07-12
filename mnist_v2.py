import tensorflow as tf
import utils.input_data as input_data

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

batch_size = 100
learning_rate = 0.8
learning_rate_decay = 0.999
max_steps = 30000

training_step = tf.Variable(0, trainable=False)


def hidden_layer(inputs, weight1, bias1, weight2, bias2):
    layer1 = tf.nn.relu(tf.matmul(inputs, weight1) + bias1)
    return tf.matmul(layer1, weight2) + bias2


x = tf.placeholder("float", [None, 784], "x")
y_ = tf.placeholder("float", [None, 10], "y_")

w1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[500]))
w2 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[10]))

y = hidden_layer(x, w1, b1, w2, b2)

average_class = tf.train.ExponentialMovingAverage(0.99, training_step)
average_op = average_class.apply(tf.trainable_variables())

average_y = hidden_layer(x, average_class.average(w1), average_class.average(b1), average_class.average(w2),
                         average_class.average(b2))

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

regularizer = tf.contrib.layers.l2_regularizer(0.0001)
regularization = regularizer(w1) + regularizer(w2)
loss = tf.reduce_mean(cross_entropy) + regularization

learning_rate = tf.train.exponential_decay(learning_rate, training_step, mnist.train.num_examples / batch_size,
                                           learning_rate_decay)

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=training_step)
train_op = tf.group(train, average_op)

correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 最多占gpu资源的70%
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
# 开始不会给tensorflow全部gpu资源 而是按需增加
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    # 准备验证数据
    validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
    # 准备测试数据
    test_feed = {x: mnist.test.images, y_: mnist.test.labels}

    for i in range(max_steps):

        if i % batch_size == 0:
            validate_accuracy = sess.run(accuracy, feed_dict=validate_feed)
            print("After %d trainging step(s) ,validation accuracy"
                  "using average model is %g%%" % (i, validate_accuracy * 100))

        xs, ys = mnist.train.next_batch(batch_size=batch_size)
        sess.run(train_op, feed_dict={x: xs, y_: ys})
        # sess.run(train, feed_dict={x: xs, y_: ys})

    test_accuracy = sess.run(accuracy, feed_dict=test_feed)
    print("After %d trainging step(s) ,test accuracy using average"
          " model is %g%%" % (max_steps, test_accuracy * 100))
