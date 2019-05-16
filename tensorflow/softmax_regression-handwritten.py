#! /usr/bin/python
#coding=utf-8

import tensorflow as tf
# import input_data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/handwritten-mnist/", one_hot=True)

# None表示此张量的第一个维度可以是任何长度的
# x不是一个特定的值，而是一个占位符placeholder，我们在TensorFlow运行计算时输入这个值。
x = tf.placeholder("float", [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 构造模型
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder("float", [None, 10])

# 损失
cross_entropy = -tf.reduce_sum(y_ * tf.log(y)) 

# train
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()

# 它能让你在运行图的时候，插入一些计算图
# 如果你没有使用InteractiveSession，那么你需要在启动session之前构建整个计算图，然后启动该计算图
# sess = tf.InteractiveSession()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))





