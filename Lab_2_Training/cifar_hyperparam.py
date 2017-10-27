############################################################
#                                                          #
#  Code for Lab 2: Intro to TensorFlow and Blue Crystal 4  #
#                                                          #
############################################################

"""Based on TensorFLow's tutorial: A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'CIFAR10'))
import cifar10 as cf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data-dir', os.getcwd() + '/dataset/',
                            'Directory where the dataset will be stored and checkpoint. (default: %(default)s)')
tf.app.flags.DEFINE_integer('max-steps', 10000,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('log-frequency', 10,
                            'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
tf.app.flags.DEFINE_integer('save-model', 1000,
                            'Number of steps between model saves (default: %(default)d)')

# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('batch-size', 128, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_float('learning-rate', 1e-3, 'Number of examples to run. (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-width', 32, 'Image width (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-height', 32, 'Image height (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-channels', 3, 'Image channels (default: %(default)d)')
tf.app.flags.DEFINE_integer('num-classes', 10, 'Number of classes (default: %(default)d)')
tf.app.flags.DEFINE_string('log-dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')


run_log_dir = os.path.join(FLAGS.log_dir,
                           'exp_bs_{bs}_lr_{lr}'.format(bs=FLAGS.batch_size,
                                                        lr=FLAGS.learning_rate))
# limit the process memory to a third of the total gpu memory
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)

# the initialiser object implementing Xavier initialisation
# we will generate weights from the uniform distribution
xavier_initializer = tf.contrib.layers.xavier_initializer(uniform=True)

def deepnn(x):
    """deepnn builds the graph for a deep net for classifying CIFAR10 images.

    Args:
        x: an input tensor with the dimensions (N_examples, 3072), where 3072 is the
          number of pixels in a standard CIFAR10 image.

    Returns:
        A tuple (y, img_summary)

        y: is a tensor of shape (N_examples, 10), with values
          equal to the logits of classifying the object images into one of 10 classes
          (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

        img_summary: a string tensor containing sampled input images.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - it would be 1 one for a grayscale image,
    # 3 for an RGB image, 4 for RGBA, etc.

    x_image = tf.reshape(x, [-1, FLAGS.img_width, FLAGS.img_height, FLAGS.img_channels])
    img_summary = tf.summary.image('Input_images',x_image)

    # Small epsilon value for the BN transform
    epsilon = 1e-3

    # First convolutional layer - maps one RGB image to 32 feature maps.
    with tf.variable_scope("Conv_1"):
        W_conv1 = weight_variable([5, 5, FLAGS.img_channels, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

        # Pooling layer - downsamples by 2X.
        h_pool1 = max_pool_2x2(h_conv1)


        # PRACTICAL 2.4:
        # Comment out the above code replacing it with the code below

        # # BATCH NORMALISATION
        # W_conv1 = weight_variable([5, 5, FLAGS.img_channels, 32])
        #
        # # Note that pre-batch normalisation bias 'b_conv1' is ommitted
        # # The effect of this bias would be eliminated when subtracting the batch mean
        # # Instead, the role of the bias is performed by the new beta variable
        #
        # # Also, note that we will apply the relu activation function later to the
        # # normalised batch
        # z1 = conv2d(x_image, W_conv1)
        #
        # # Calculate batch mean and variance
        # batch_mean1, batch_var1 = tf.nn.moments(z1, [0])
        #
        # # Apply the initial batch normalizing transform
        # z1_hat = (z1 - batch_mean1) / tf.sqrt(batch_var1 + epsilon)
        #
        # # Create two new parameters, scale and shift (gamma and beta)
        # gamma1 = tf.Variable(tf.ones([32]))
        # beta1 = tf.Variable(tf.zeros([32]))
        #
        # # Scale and shift to obtain the final output of the batch normalisation
        # # this value is fed into the activation function (here a relu)
        # BN1 = gamma1 * z1_hat + beta1
        #
        # h_conv1_bn = tf.nn.relu(BN1)
        #
        # # Pooling layer - downsamples by 2X.
        # h_pool1 = max_pool_2x2(h_conv1_bn)

    with tf.variable_scope("Conv_2"):
        # Second convolutional layer -- maps 32 feature maps to 64.
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        # Second pooling layer.
        h_pool2 = max_pool_2x2(h_conv2)

        # PRACTICAL 2.4:
        # Comment out the above code replacing it with the code below

        # # BATCH NORMALISATION
        # # Second convolutional layer -- maps 32 feature maps to 64.
        # W_conv2 = weight_variable([5, 5, 32, 64])

        # z2 = conv2d(h_pool1, W_conv2)

        # # Calculate batch mean and variance
        # batch_mean2, batch_var2 = tf.nn.moments(z2, [0])

        #  # Create two new parameters, scale and shift (gamma and beta)
        # gamma2 = tf.Variable(tf.ones([64]))
        # beta2 = tf.Variable(tf.zeros([64]))

        # # Using Tensorflow built-in BN function this time
        # BN2 = tf.nn.batch_normalization(z2, batch_mean2, batch_var2, beta2, gamma2, epsilon)
        # h_conv2_bn = tf.nn.relu(BN2)

        # # Second pooling layer
        # h_pool2 = max_pool_2x2(h_conv2_bn)

    with tf.variable_scope("FC_1"):
        # Fully connected layer 1 -- after 2 round of downsampling, our 32x32 image
        # is down to 8x8x64 feature maps -- maps this to 1024 features.
        W_fc1 = weight_variable([8 * 8 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.variable_scope("FC_2"):
        # Map the 1024 features to 10 classes
        W_fc2 = weight_variable([1024, FLAGS.num_classes])
        b_fc2 = bias_variable([FLAGS.num_classes])

        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    return y_conv, img_summary


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name='convolution')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name='pooling')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)

    # PRACTICAL 2.6.1: Try using different initialisation techniques:
    #initial = tf.zeros(shape)
    #initial = tf.ones(shape)

    return tf.Variable(initial, name='weights')
    # PRACTICAL 2.6.2: Try using xavier initialisation:
    #return tf.Variable(xavier_initializer(shape))


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='biases')
    # PRACTICAL 2.6.2: Try using xavier initialisation:
    #return tf.Variable(xavier_initializer(shape))


def main(_):
    tf.reset_default_graph()

    cifar = cf.cifar10(batchSize=FLAGS.batch_size, downloadDir=FLAGS.data_dir)

    with tf.variable_scope("inputs"):
        x = tf.placeholder(tf.float32, [None, FLAGS.img_width * FLAGS.img_height * FLAGS.img_channels])
        y_ = tf.placeholder(tf.float32, [None, FLAGS.num_classes])

    # Build the graph for the deep net
    y_conv, img_summary = deepnn(x)

    with tf.variable_scope("x_entropy"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    global_step = tf.Variable(0, trainable=False)  # this will be incremented automatically by tensorflow
    decay_steps = 1000  # decay the learning rate every 1000 steps
    decay_rate = 0.8  # the base of our exponential for the decay

    decayed_learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                               decay_steps, decay_rate, staircase=True)

    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
    # PRACTICAL 2.3: Replace the line above with the one commented out below
    #train_step = tf.train.AdamOptimizer(decayed_learning_rate).minimize(cross_entropy, global_step=global_step)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    loss_summary = tf.summary.scalar("Loss", cross_entropy)
    acc_summary = tf.summary.scalar("Accuracy", accuracy)
    learning_rate_summary = tf.summary.scalar("Learning Rate", decayed_learning_rate)

    # summaries for TensorBoard visualisation
    validation_summary = tf.summary.merge([img_summary, acc_summary])
    training_summary = tf.summary.merge([img_summary, loss_summary, learning_rate_summary])
    test_summary = tf.summary.merge([img_summary, acc_summary])

    # saver for checkpoints
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        summary_writer = tf.summary.FileWriter(run_log_dir + "_train", sess.graph)
        summary_writer_validation = tf.summary.FileWriter(run_log_dir + "_validate", sess.graph)

        sess.run(tf.global_variables_initializer())

        # Training and validation
        for step in range(FLAGS.max_steps):
            # Training: Backpropagation using test set
            (trainImages, trainLabels) = cifar.getTrainBatch()
            (testImages, testLabels) = cifar.getTestBatch()

            _, summary_str = sess.run([train_step, training_summary], feed_dict={x: trainImages, y_: trainLabels})

            if step % FLAGS.log_frequency == 0:
                summary_writer.add_summary(summary_str, step)

            # Validation: Monitoring accuracy using validation set
            if step % (FLAGS.log_frequency + 1) == 0:
                validation_accuracy, summary_str = sess.run([accuracy, validation_summary],
                                                            feed_dict={x: testImages, y_: testLabels})
                print('step {}, accuracy on validation set : {}'.format(step, validation_accuracy))
                summary_writer_validation.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % FLAGS.save_model == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(run_log_dir + "_train", 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        # Resetting the internal batch indexes
        cifar.reset()
        evaluated_images = 0
        test_accuracy = 0
        batch_count = 0

        while evaluated_images != cifar.nTestSamples:
            # Don't loop back when we reach the end of the test set
            (testImages, testLabels) = cifar.getTestBatch(allowSmallerBatches=True)
            test_accuracy_temp, _ = sess.run([accuracy, test_summary], feed_dict={x: testImages, y_: testLabels})

            batch_count = batch_count + 1
            test_accuracy = test_accuracy + test_accuracy_temp
            evaluated_images = evaluated_images + testLabels.shape[0]

        test_accuracy = test_accuracy / batch_count
        print('test set: accuracy on test set: %0.3f' % test_accuracy)


if __name__ == '__main__':
    tf.app.run(main=main)
