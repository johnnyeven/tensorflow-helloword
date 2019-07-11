import tensorflow as tf
from sklearn.datasets import load_boston, make_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

train_steps = 10000
learn_rate = 0.05
stddev = 0.01

W1 = tf.Variable(tf.random_normal([13, 6], dtype=tf.float64, stddev=stddev))
b1 = tf.Variable(tf.zeros([6, ], dtype=tf.float64))
W2 = tf.Variable(tf.random_normal([6, 1], dtype=tf.float64, stddev=stddev))
b2 = tf.Variable(tf.zeros([1], dtype=tf.float64))


def inputs():
    boston = load_boston()
    feature_X = boston.data
    expected_Y = boston.target
    # feature_X, expected_Y = make_regression(n_samples=1000, n_features=13, noise=1.5)
    # 标准化数据
    # scaler = StandardScaler()
    # scaler.fit(boston.data)
    # x = scaler.transform(boston.data)
    # target = boston.target.reshape(-1, 1)
    # y = target

    scaler = MinMaxScaler()
    x = scaler.fit_transform(feature_X)
    target = expected_Y.reshape(-1, 1)
    y = scaler.fit_transform(target)

    return x, y


# 两层计算模型
def inference(x):
    a = tf.nn.relu(tf.matmul(x, W1) + b1)
    return tf.matmul(a, W2) + b2


# 均方差损失函数，用以评判预测值与目标值之间的差距大小
def loss(y, y_):
    return tf.reduce_mean(tf.squared_difference(y_, y))


def train(learn_rate, loss_func):
    return tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(loss_func)


def visualization(y, y_):
    plt.figure()
    plt.plot(y, 'bo', alpha=0.3)
    plt.plot(y_, 'ro', alpha=0.3)
    plt.ylabel('price')
    plt.show()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    expected_x, expected_y = inputs()
    batch_size = len(expected_y)

    predict_y = inference(expected_x)
    loss_func = loss(predict_y, expected_y)
    train_func = train(learn_rate, loss_func)

    for i in range(train_steps):
        sess.run(train_func)
        if i % 10 == 0:
            print(i, sess.run(loss_func))

    y = sess.run(predict_y)
    visualization(y, expected_y)
