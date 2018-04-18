import os.path
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

#Define parameter
learning_rate = 0.01
train_epoch = 100
batch_size = 100

def run_training():
    # Step1: Read data
    mnist = input_data.read_data_sets('data/mnist', one_hot=True)

    # Step 2: Create placeholders for features and labels
    X = tf.placeholder(tf.float32, shape=[None, 784], name="X")
    Y = tf.placeholder(tf.float32, shape=[None, 10], name="Y")

    # Step 3: create variable W, b
    W = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name="Weight")
    b = tf.Variable(tf.zeros([1, 10]), name="Bias")

    # Step4: Build model
    logits = tf.matmul(X, W) + b

    # Step5: define loss
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name="loss")
    loss = tf.reduce_mean(entropy)

    # Step6: define training op
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    with tf.Session() as sess:
        # To visualize using tensorboard
        start_time = time.time()
        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter("./graph/logistic_reg", sess.graph)
        n_batches = int(mnist.train.num_examples / batch_size)

        for i in range(train_epoch):
            total_loss = 0

            for j in range(n_batches):
                X_batch, Y_batch = mnist.train.next_batch(batch_size)
                op, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})
                total_loss += loss_batch
            print('Average loss epoch {0}:{1}'.format(i, total_loss / n_batches))

        print('Total time: {0} seconds'.format(time.time() - start_time))
        print('Optimization Finished!')

        # Evaluate Model
        correct_prediction = tf.equal(tf.arg_max(Y, 1), tf.arg_max(logits, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

def main(_):
    run_training()
if __name__=='__main__':
    tf.app.run()
