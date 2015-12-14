import numpy as np
import tensorflow as tf
import itertools

if __name__ == '__main__':

    hiddenDim = 10

    g = np.matrix(([1, 0, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0, 1],
                   [0, 0, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]), dtype=np.float32)

    x_data = np.array(list(itertools.product([0, 1], repeat=4)), dtype=np.float32)
    y_data = [np.squeeze(np.asarray(np.dot(i, g) % 2)) for i in x_data]

    x = tf.placeholder(tf.float32, shape=[None, 4], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 7], name="y_")

    b = tf.Variable(tf.zeros([hiddenDim]), name="b")
    W = tf.Variable(tf.random_uniform([4, hiddenDim], -0.5, 0.5),
                    name="W")

    b0 = tf.Variable(tf.zeros([1]))
    W0 = tf.Variable(tf.random_uniform([hiddenDim, 7], -0.5, 0.5))

    hidden = tf.sigmoid(tf.matmul(x, W) + b)

    y = tf.sigmoid(tf.matmul(hidden, W0) + b0)

    loss = tf.reduce_mean(tf.square(y_ - y))
    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    init = tf.initialize_all_variables()
    sess = tf.Session()  # create the session and therefore the graph
    sess.run(init)  # initialize all variables

    for epoch in xrange(0, 25001):
        run = sess.run([train, loss, W, b, W, b0],
                       feed_dict={x: x_data, y_: y_data})
        if epoch % 1000 == 0:
            print epoch, run

            print("Loss is %s \n " % run[1])

    for i in x_data:
        print("Input was %s" % i)
        output = np.array(sess.run(y, feed_dict={x: [i]}))
        print("Output is %s" % output)
        print("Rounded is %s \n \n" % np.round(output))
