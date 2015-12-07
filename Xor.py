# TODO Work on a multilayer perceptron for XOR
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    x_data = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    y_data = np.array([[-1.0], [-1.0], [1.0], [1.0]], dtype=np.float32)

    hiddenDim = 2
    # x = tf.placeholder(tf.float32, [4, 2])
    # y_ = tf.placeholder(tf.float32, [4, 1])

    b = tf.Variable(tf.zeros([hiddenDim]))
    W = tf.Variable(tf.random_uniform([2, hiddenDim], -0.5, 0.5))

    b0 = tf.Variable(tf.zeros([1]))
    W0 = tf.Variable(tf.random_uniform([hiddenDim, 1], -0.5, 0.5))

    hidden = tf.nn.softmax(tf.matmul(x_data, W) + b)

    y = tf.nn.softmax(tf.matmul(hidden, W0) + b0)

    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.02)
    train = optimizer.minimize(loss)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    with sess.as_default():
        for step in xrange(0, 40001):
            train.run()
            if step % 500 == 0:
                print step, loss.eval(x_data.all(), sess)
