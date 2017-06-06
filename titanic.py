import tensorflow as tf
import numpy as np
import csv

file_name = "resources/titanic-data/train.csv"
with open(file_name, 'r') as f:
    print(f.readline())

train_data = np.genfromtxt(file_name, delimiter=",", skip_header=1, dtype=str)

print(train_data.shape)

# for row in train_data:
#     print(row[6])

expected = np.reshape(np.array(train_data[:, 1], dtype=np.int8), (891, 1))
#
input_sex = np.array([1 if i == 'female' else 0 for i in train_data[:, 5]], dtype=np.int)

input_age = np.array([28 if i == '' else i for i in train_data[:, 6]], dtype=np.float)

input = np.transpose(np.array([input_sex, input_age]))

print(expected.shape)
print(input.shape)

# Build graph

x = tf.placeholder(tf.float32, shape=[891, 2])
W = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [891, 1])

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

# for _ in range(1000):
#   batch_xs, batch_ys = mnist.train.next_batch(100)
sess.run(train_step, feed_dict={x: input, y_: expected})

# test model
correct_prediction = tf.equal(y, y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))




