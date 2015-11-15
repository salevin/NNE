import numpy as np

import tensorflow as tf

# Just mucking about, needs a huge change

if __name__ == '__main__':
    # Make 100 binary data-points in NumPy.
    x_data = np.array([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=np.float32)
    y_data = np.array([1, -1, -1, 1], dtype=np.float32)

    # Construct a linear model.
    b = tf.Variable(tf.zeros([1]))
    W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
    y = tf.matmul(W, x_data) + b

    # Minimize the squared errors.
    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    # For initializing the variables.
    init = tf.initialize_all_variables()

    # Launch the graph
    sess = tf.Session()
    sess.run(init)

    # Fit the plane.
    for step in xrange(0, 201):
        sess.run(train)
        if step % 20 == 0:
            print step, sess.run(W), sess.run(b)
