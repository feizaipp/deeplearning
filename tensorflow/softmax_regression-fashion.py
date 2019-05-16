#! /usr/bin/python
#coding=utf-8

import gzip
import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from PIL import Image

mnist = input_data.read_data_sets("../data/fashion-mnist/", one_hot=True)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def load_data(): 
    base = "../data/fashion-mnist/"
    files = ['train-labels-idx1-ubyte.gz',
        'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz']
    paths = []
    for fname in files:
        paths.append(os.path.join(base, fname))
    with gzip.open(paths[0], 'rb') as lbpath:
        # np.uint8: Unsigned integer (0 to 255);
        # offset =8:Start reading the buffer from this offset (in bytes)
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[1], 'rb') as imgpath:
        # offset = 16;np.frombuffer: Interpret a buffer as a
        # 1-dimensional array;np.reshape(28,28):
        # Gives a new shape to an array without changing its data
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer( imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
    return (x_train, y_train), (x_test, y_test) 

def make_one_hot(data):
    return (np.arange(10)==data[:,None]).astype(np.integer)

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = np.array(indices[i: min(i + batch_size, num_examples)])
        yield np.take(features, j, axis=0), np.take(labels, j, axis=0)

def show_fashion_mnist(images, labels):
    pass

(train_images, train_labels), (test_images, test_labels) = load_data()

train_images = np.reshape(train_images, [-1, 784])
#train_labels = tf.one_hot(train_labels, 10, 1, 0)
train_labels = make_one_hot(train_labels)

test_images = np.reshape(test_images, [-1, 784])
#test_labels = tf.one_hot(test_labels, 10, 1, 0)
test_labels = make_one_hot(test_labels)

x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder("float", [None, 10])

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    if i % 100 == 0:
        print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))





