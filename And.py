import numpy as np

if __name__ == '__main__':
    # Make a single epoch of AND
    x_data = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]], dtype=np.int)
    y_data = np.array([1, -1, -1, -1], dtype=np.int)

    weights = np.zeros([3], dtype=np.int)

    epochs = input("Number of epochs (Try 10): ")

    bias = .2

    learning_rate = 1

    for i in range(epochs):
        for z in range(len(x_data[0])):
            target = y_data[z]
            x = x_data[:, z]
            y_in = np.dot(x, weights)
            y = 1 if y_in > bias else (0 if -bias <= y_in <= bias else -1)

            if y != target:
                change = learning_rate * x * target
                weights = np.add(weights, change)

    print("Weights are %s" % weights)

    go_on = True
    while go_on:
        x1 = input("First binary input: ")
        x2 = input("Second binary input: ")

        x = np.array([x1, x2, 1])

        y_in = np.dot(x, weights)
        y = 1 if y_in > bias else (0 if -bias <= y_in <= bias else -1)

        print(y)

        want = raw_input("Want to continue? (y/n): ")

        chosen = False
        while not chosen:
            if want.lower() == 'n':
                go_on = False
                chosen = True
            elif want.lower() != 'y':
                want = raw_input("What was that? (y/n): ")
            else:
                chosen = True
