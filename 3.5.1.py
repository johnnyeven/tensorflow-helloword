import tensorflow as tf

x = tf.constant([[1.0, 2.0]])

w1 = tf.Variable(tf.random_normal([2, 3], seed=1, stddev=1))
w2 = tf.Variable(tf.random_normal([3, 1], seed=1, stddev=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(y))

writer = tf.summary.FileWriter('./', tf.get_default_graph())
writer.close()
