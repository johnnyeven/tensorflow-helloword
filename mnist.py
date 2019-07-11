import tensorflow as tf
import utils.input_data as input_data

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

batch_size = 100
learning_rate = 0.8
learning_rate_decay = 0.999
max_steps = 30000


def hidden_layer(inputs, weight1, bias1, weight2, bias2):
    layer1 = tf.nn.relu(tf.matmul(inputs, weight1) + bias1)
    return tf.matmul(layer1, weight2) + bias2


x = tf.placeholder("float", [None, 784], "x")
y_ = tf.placeholder("float", [None, 10], "y_")

training_step = tf.Variable(0, trainable=False)

W1 = tf.Variable(tf.truncated_normal([784, 300]))
b1 = tf.Variable(tf.constant(0.1, shape=[300]))
W2 = tf.Variable(tf.truncated_normal([300, 10]))
b2 = tf.Variable(tf.constant(0.1, shape=[10]))

y = hidden_layer(x, W1, b1, W2, b2)

# average_class = tf.train.ExponentialMovingAverage(0.99, training_step)
# average_op = average_class.apply(tf.trainable_variables())
#
# average_y = hidden_layer(x, average_class.average(W1), average_class.average(b1), average_class.average(W2),
#                          average_class.average(b2))

regularizer = tf.contrib.layers.l2_regularizer(0.0001)
regularization = regularizer(W1) + regularizer(W2)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
loss = tf.reduce_mean(cross_entropy) + regularization

learning_rate = tf.train.exponential_decay(learning_rate, training_step, mnist.train.num_examples / batch_size,
                                           learning_rate_decay)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step=training_step)
train_op = tf.group(train)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(max_steps):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_op, feed_dict={x: batch_xs, y_: batch_ys})

        if i % 100 == 0:
            print(i, sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys}),
                  sess.run(accuracy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels}) * 100)

    # 保存训练模型
    # saver = tf.train.Saver()
    # save_path = saver.save(sess, "models/mnist_01.ckpt")

# 使用保存的训练模型
# saver = tf.train.Saver()
# saver.restore(sess, "D:\sample\model.ckpt")
# result = sess.run(y, feed_dict={x: data})
