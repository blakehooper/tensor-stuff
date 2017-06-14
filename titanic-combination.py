import numpy as np
import pandas as pd
import tensorflow as tf

def buildSuccess(x):
  result = np.ones((2))
  result[int(x)] = 0
  return result

def readData(fileName):
  train_data = pd.read_csv(fileName)

  train_data["Age"] = train_data["Age"].fillna(train_data["Age"].median())

  train_data.loc[train_data["Sex"] == "male", "Sex"] = 0
  train_data.loc[train_data["Sex"] == "female", "Sex"] = 1

  train_data["Embarked"] = train_data["Embarked"].fillna("S")

  train_data.loc[train_data["Embarked"] == "S", "Embarked"] = 0
  train_data.loc[train_data["Embarked"] == "C", "Embarked"] = 1
  train_data.loc[train_data["Embarked"] == "Q", "Embarked"] = 2

  predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

  return train_data[predictors], np.array([buildSuccess(i) for i in train_data["Survived"]])

train_data, train_expected = readData("./resources/titanic-data/train.csv")
#test_data, test_expected = readData("./resources/titanic-data/test.csv")

def accuracy(produced, provided):
  return (100.0 * np.sum(np.argmax(produced, 1) == np.argmax(provided, 1))
          / produced.shape[0])

graph = tf.Graph()
with graph.as_default():

  #test_dataset = tf.constant(test_data)

  ## Input layer
  # an input size 891, 7
  train_dataset = tf.placeholder(tf.float32, shape=(891, 7))
  # an input size 891, 2
  expected_placeholder = tf.placeholder(tf.float32, shape=(891, 2))

  ## Hidden layer 1
  hidden_nodes = 1024
  # Trainable size 7 X 1024
  hidden_weights = tf.Variable(tf.truncated_normal([7, hidden_nodes]))
  # Trainable size 1 X 1024
  hidden_biases = tf.Variable(tf.zeros([hidden_nodes]))

  hidden_layer = tf.nn.relu(tf.matmul(train_dataset, hidden_weights) + hidden_biases)

  # Output layer
  # Trainable size 1024 x 1
  weights = tf.Variable(tf.truncated_normal([1024, 2]))
  # Trainable size 1 X 1
  biases = tf.Variable(tf.zeros([2]))

  logits = tf.matmul(hidden_layer, weights) + biases

  #Write logits as probabilites
  train_prediction = tf.nn.softmax(logits)

  # Define loss and optimizer
  # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=train_expected))
  beta = 0.01
  loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=train_expected)
  loss = tf.reduce_mean(loss + beta * tf.nn.l2_loss(hidden_weights) + beta * tf.nn.l2_loss(weights))
  optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

  # test1 = tf.nn.relu(tf.matmul(test_dataset, hidden_weights) + hidden_biases)
  # test_prediction = tf.nn.softmax(tf.matmul(test1, weights) + biases)

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  for x in range(20):
    feed_dict = {train_dataset: train_data, expected_placeholder: train_expected }
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    print("train batch accuracy: %.1f%%" % accuracy(predictions, train_expected))

