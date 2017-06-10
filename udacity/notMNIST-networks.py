# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory

image_size = 28
num_labels = 10


def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
  return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


batch_size = 128

graph = tf.Graph()
with graph.as_default():

  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  ## Input layer
  # an input size 128, 784
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
  # an input size 128, 10
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))


  ## Hidden layer 1
  hidden_nodes = 1024
  # Trainable size 784 X 1024
  hidden_weights = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_nodes]))
  # Trainable size 1 X 1024
  hidden_biases = tf.Variable(tf.zeros([hidden_nodes]))

  hidden_layer = tf.nn.sigmoid(tf.matmul(tf_train_dataset, hidden_weights) + hidden_biases)

  ## Hidden layer 2
  hidden_nodes2 = 512
  # Trainable size 1024 X 512
  hidden_weights2 = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes2]))
  # Trainable size 1 X 512
  hidden_biases2 = tf.Variable(tf.zeros([hidden_nodes2]))

  hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer, hidden_weights2) + hidden_biases2)

  # Output layer
  # Trainable size 512 X 10
  weights = tf.Variable(tf.truncated_normal([hidden_nodes2, num_labels]))
  # Trainable size 1 X 10
  biases = tf.Variable(tf.zeros([num_labels]))

  logits = tf.matmul(hidden_layer2, weights) + biases

  #Write logits as probabilites
  train_prediction = tf.nn.softmax(logits)

  beta = 0.01
  # Define loss and optimizer
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
  #loss = tf.reduce_mean(loss + beta * tf.nn.l2_loss(hidden_weights) + beta * tf.nn.l2_loss(weights))
  optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

  # Use determined weights and biases to verify validation and test data
  valid1 = tf.nn.sigmoid(tf.matmul(tf_valid_dataset, hidden_weights) + hidden_biases)
  valid2 = tf.nn.relu(tf.matmul(valid1, hidden_weights2) + hidden_biases2)
  valid_prediction = tf.nn.softmax(tf.matmul(valid2, weights) + biases)

  test1 = tf.nn.sigmoid(tf.matmul(tf_test_dataset, hidden_weights) + hidden_biases)
  test2 = tf.nn.relu(tf.matmul(test1, hidden_weights2) + hidden_biases2)
  test_prediction = tf.nn.softmax(tf.matmul(test2, weights) + biases)

num_steps = 6001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))