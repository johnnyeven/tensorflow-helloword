import tensorflow as tf

x = tf.constant([0.9, 0.85], shape=[1, 2])

w1 = tf.Variable(tf.constant([
    [0.2, 0.1, 0.3],
    [0.2, 0.4, 0.3],
], shape=[2, 3]), name="w1")

w2 = tf.Variable(tf.constant([
    [0.2, 0.5, 0.25],
], shape=[3, 1]), name="w2")

b1 = tf.constant([-0.3, 0.1, 0.2], shape=[1, 3], name="b1")
b2 = tf.constant([-0.3], shape=[1], name="b2")

y_ = tf.constant([0.2], shape=[1], name="y_")

init_op = tf.global_variables_initializer()

# 第一层
a = tf.nn.relu(tf.matmul(x, w1) + b1)
# 第二层
y = tf.nn.relu(tf.matmul(a, w2) + b2)
# 求交叉熵
ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(y))
    print(sess.run(ce))
