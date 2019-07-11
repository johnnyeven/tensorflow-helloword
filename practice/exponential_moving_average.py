import tensorflow as tf
import matplotlib.pyplot as plt

max_steps = 1000


def visualization(y, y_):
    plt.figure()
    plt.plot(y, 'b', alpha=0.5)
    plt.plot(y_, 'r', alpha=0.5)
    plt.show()


# 定义一个变量用于计算滑动平均,这个变量初始值为0,这里手动制定了变量
var = tf.random_normal([max_steps], stddev=0.1)
var1 = tf.Variable(tf.zeros([max_steps], dtype=tf.float32), trainable=False)

v1 = tf.Variable(0, dtype=tf.float32)  # ---------------------------------------[1]
# step变量模拟神经网络中迭代的轮数,可以用于动态控制衰减率  step-->num_update
step = tf.Variable(0, trainable=False)

# 定义一个滑动平均的类(class).初始化时给定了衰减率(0.99)和控制衰减率的变量step
Moving_average = tf.train.ExponentialMovingAverage(0.99, step)
# 定义一个更新变量滑动平均的操作
# 这里需要给定一个列表,每次执行这个操作时,这个列表中的变量都会被更新

# 更新v1
maintain_averages_op = Moving_average.apply([v1])

with tf.Session() as sess:
    # 初始化所有变量
    sess.run(tf.global_variables_initializer())

    # 通过Moving_average.average(v1)获取滑动平均之后的取值
    # 在初始化之后变量v1的值和v1的滑动平均都为0

    for i in range(max_steps):
        avg = sess.run(Moving_average.average(v1))
        print(i)
        sess.run(v1.assign(var[i]))
        sess.run(tf.assign(var1[i], avg))
        sess.run(maintain_averages_op)

    visualization(sess.run(var), sess.run(var1))
