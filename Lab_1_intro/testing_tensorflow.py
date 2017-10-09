############################################################
#                                                          #
#  Code for Lab 1: Intro to TensorFlow and Blue Crystal 4  # 
#                                                          #
############################################################

"""Based on TensorFLow's tutorial: A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', os.getcwd() + '/dataset/',
                           """Directory where the dataset will be stored """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir', os.getcwd() + '/logs/experiment_bs_100',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 2000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """Number of steps to log results to the console and save summaries""")
tf.app.flags.DEFINE_integer('save_model', 1000,
                            """Number of steps for saving the model periodically""")


# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('batch_size', 100,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 
	                        """Number of examples to run.""")

toolbar_width = 20




def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.

  x_image = tf.reshape(x, [-1, 28, 28, 1])

  img_summary = tf.summary.image('Input_images',x_image)

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.variable_scope("Conv_1") as scope:
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    h_pool1 = max_pool_2x2(h_conv1)

  with tf.variable_scope("Conv_2") as scope:

    # Second convolutional layer -- maps 32 feature maps to 64.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    
    # Second pooling layer.
    h_pool2 = max_pool_2x2(h_conv2)

  with tf.variable_scope("FC_1") as scope:

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  with tf.variable_scope("FC_2") as scope:  

    # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
  return y_conv, img_summary


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name='convolution')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  #"name: name for operation"
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name='pooling')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name='weights')


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name='biases')


def main(_):


  tf.reset_default_graph()
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  with tf.variable_scope("inputs") as scope:

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net
  y_conv, img_summary = deepnn(x)

  with tf.variable_scope("x_entropy") as scope:

    cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  
  train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')

  loss_summary =  tf.summary.scalar("Loss", cross_entropy)
  
  acc_summary =  tf.summary.scalar("Accuracy", accuracy)
  
  # summaries for TensorBoard visualisation

  validation_summary = tf.summary.merge([img_summary,acc_summary])
  training_summary = tf.summary.merge([img_summary,loss_summary]) 
  test_summary = tf.summary.merge([img_summary,acc_summary])
  
  # saver for checkpoints
  saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

  with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    # For loop for train and validation

    # setup toolbar
    print ('\033[94m' + "Running test: " )
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

    for step in range(FLAGS.max_steps):

      # Training -----------------------------------------------------------------------------------------------------------------------------------
      try:
          # Backpropagation using test set

        batch = mnist.train.next_batch(FLAGS.batch_size)

        _,summary_str = sess.run([train_step, training_summary], feed_dict={x: batch[0], y_: batch[1]})

        if step % FLAGS.log_frequency == 0:
          sys.stdout.write("-")
          sys.stdout.flush()
     

      except Exception as e: 

        print('\033[0m')
        print ( e)
        print ( '\033[0m')
      # --- Train done
    # Testing -------------------------------------------------------------------------------------------------------------------------------------- 
    # Accuracy on TEST set
    try:
      print ( '\033[0m')
      test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images[0:100], y_: mnist.test.labels[0:100]})
      if test_accuracy > 0.8 : print('\033[92m' + "Test completed, everything looks good!" + '\033[0m') 
      tf.gfile.DeleteRecursively(FLAGS.data_dir)
    except Exception as e: 
      print ('\033[93m')
      print ( e)
      print ( '\033[0m')

if __name__ == '__main__':

  tf.app.run(main=main)