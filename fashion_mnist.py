import tensorflow as tf
import numpy as np
import os


# ====================================== Utilities =================================== #

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
        labels = np.eye(labels.max() + 1)[labels]  # OneHotEncode labels.

    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        images = images / 255.0  # Normalize data.

    return images, labels


def get_data():
    #  get the train,val and test sets
    path_to_fmnist_folder = './mnist_data'

    x_train, y_train = load_mnist(path_to_fmnist_folder, kind='train')
    x_test, y_test = load_mnist(path_to_fmnist_folder, kind='t10k')

    d = x_train.shape[1]
    num_classes = y_train.shape[1]

    return x_train, x_test, y_train, y_test, d, num_classes


def split_to_batches(l, n):
    n = max(1, n)
    return (l[i:i + n] for i in xrange(0, len(l), n))


# ==================================================================================== #

def main():
    # ================================================================================ #
    # Hyper parameters.
    # ================================================================================ #
    learning_rate = 0.1
    num_epochs = 50
    batch_size = 100

    # ================================================================================ #
    # Read Data.
    # ================================================================================ #
    x_train, x_test, y_train, y_test, d, num_classes = get_data()

    # ================================================================================ #
    # Build Graph and Define optimizers.
    # ================================================================================ #
    X = tf.placeholder(tf.float32, shape=(None, d), name="X")
    W = tf.Variable(tf.random_normal(shape=(d, num_classes), stddev=0.1), name="W")
    b = tf.Variable(tf.random_uniform(minval=-0.01, maxval=0.01, shape=(1, num_classes)), name="b")

    y = tf.placeholder(tf.float64, shape=(None, num_classes), name="y")
    y_hat = tf.matmul(X, W) + b

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_hat))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # ================================================================================ #
    # Test tensors.
    # ================================================================================ #
    labels = tf.argmax(tf.nn.softmax(y_hat), axis=1)
    predictions = tf.argmax(y, axis=1)
    correct_prediction = tf.equal(labels, predictions)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

    # ================================================================================ #
    # Initialize Tensorflow objects.
    # ================================================================================ #
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # ================================================================================ #
    # Create a summary to monitor tensors.
    # ================================================================================ #
    tf.summary.scalar("Loss", cost)
    tf.summary.scalar("Accuracy", accuracy)
    tf.summary.image("Image", tf.reshape(X, [-1, 28, 28, 1]))

    merged_summary = tf.summary.merge_all()
    
    # ================================================================================ #
    # Create and run training session.
    # ================================================================================ #
    with tf.Session() as training_sess:

        training_sess.run(init)

        writer = tf.summary.FileWriter('./tensorboard', graph=tf.get_default_graph())

        for epoch in range(num_epochs):
            chunks_X, chunks_y = split_to_batches(x_train, batch_size), split_to_batches(y_train, batch_size)

            for ii, (batch_X, batch_y) in enumerate(zip(chunks_X, chunks_y)):
                _, c, acc, summary = training_sess.run([optimizer, cost, accuracy, merged_summary],
                                                       feed_dict={X: batch_X, y: batch_y})

                print "Epoch:{}, Batch:{}, Cost:{}, Accuracy:{}".format(epoch + 1, ii + 1, c, acc)

                writer.add_summary(summary=summary, global_step=epoch)

        print "Optimization Finished!\n"

        saver.save(training_sess, os.path.join('./model'), global_step=num_epochs)

        writer.close()

    # ================================================================================ #
    # Create and run training session.
    # ================================================================================ #
    with tf.Session() as test_sess:

        saver.restore(test_sess, tf.train.latest_checkpoint('./'))

        c_train = test_sess.run(cost, feed_dict={X: x_train, y: y_train})
        acc, c_test = test_sess.run([accuracy, cost], feed_dict={X: x_test, y: y_test})

        print "Test Accuracy:{}".format(acc)
        print "Train Cost:{}".format(c_train)
        print "Test Cost:{}".format(c_test)


if __name__ == '__main__':
    main()
