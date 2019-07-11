import tensorflow as tf

a = tf.placeholder(tf.float64, 2, 'input1')
b = tf.placeholder(tf.float64, 2, 'input2')

result = a+b

with tf.Session() as sess:
    print(sess.run(result, {a: [1.0, 2.0], b: [3.0, 4.0]}))

# with placeholder
a = tf.placeholder(tf.float32, 2, 'input3')
b = tf.placeholder(tf.float32, (4, 2), 'input4')

result = a+b

with tf.Session() as sess:
    print(
        sess.run(result, {
            a: [1.0, 2.0],
            b: [
                [1.0, 2.0],
                [2.0, 3.0],
                [7.0, 1.0],
                [9.0, 14.5],
            ]
        })
    )
