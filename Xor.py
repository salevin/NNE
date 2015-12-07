# TODO Work on a multilayer perceptron for XOR
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    x_data = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                      dtype=np.float32)
    y_data = np.array([[-1.0], [-1.0], [1.0], [1.0]],
                      dtype=np.float32)

    hiddenDim = 4

    x = tf.placeholder(tf.float32, shape=[None, 2], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 1], name="y_")

    b = tf.Variable(tf.zeros([hiddenDim]), name="b")
    W = tf.Variable(tf.random_uniform([2, hiddenDim], -0.5, 0.5),
                    name="W")

    b0 = tf.Variable(tf.zeros([1]))
    W0 = tf.Variable(tf.random_uniform([hiddenDim, 1], -0.5, 0.5))

    hidden = tf.nn.softmax(tf.matmul(x, W) + b)

    y = tf.nn.softmax(tf.matmul(hidden, W0) + b0)

    entropy = tf.square(y_ - y)
    loss = tf.reduce_mean(entropy)
    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for epoch in xrange(0, 2001):
        run = sess.run([train, loss, W, b, W, b0],
                       feed_dict={x: x_data, y_: y_data})
        if epoch % 400 == 0:
            print epoch, run
