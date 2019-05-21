#! /usr/bin/python
#coding=utf-8

import tensorflow as tf
# import input_data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/handwritten-mnist/", one_hot=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

learning_rate = 0.01
num_epochs = 10
batch_size = 100
n_hidden = 256
n_output = 10
n_input = 748

x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])

W_1 = weight_variable([784, 256])
b_1 = bias_variable([256])

W_2 = weight_variable([256, 10])
b_2 = bias_variable([10])

layer_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)
layer_2 = tf.nn.softmax(tf.matmul(layer_1, W_2) + b_2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(layer_2))

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(layer_2, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
