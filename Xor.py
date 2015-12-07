# TODO Work on a multilayer perceptron for XOR

# House music and Redbull fuueeel
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

    hidden = tf.sigmoid(tf.matmul(x, W) + b)

    y = tf.sigmoid(tf.matmul(hidden, W0) + b0)

    entropy = -(y_ * tf.log(y) + (1 - y_) * tf.log(1 - y))
    loss = tf.reduce_mean(tf.square(y_ - y))
    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    init = tf.initialize_all_variables()
    sess = tf.Session()  # create the session and therefore the graph
    sess.run(init)  # initialize all variables

    for epoch in xrange(0, 100001):
        run = sess.run([train, loss, W, b, W, b0],
                       feed_dict={x: x_data, y_: y_data})
        if epoch % 400 == 0:
            print epoch, run, "\n"

            print("Erros is %s" % format(run[1]))
