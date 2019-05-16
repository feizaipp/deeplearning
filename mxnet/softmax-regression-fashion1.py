import os
import sys
import mxnet as mx
from mxnet import autograd, nd
from mxnet.gluon import data as gdata
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn, utils as gutils
import numpy as np
import gzip
from PIL import Image

def evaluate_accuracy1(test_images, test_labels, batch_size, net, ctx=[mx.cpu()]):
    """Evaluate accuracy of a model on the given data set."""
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc_sum, n = nd.array([0]), 0
    for features, labels in data_iter(batch_size, test_images, test_labels):
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc_sum += (net(X).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size
        acc_sum.wait_to_read()
    return acc_sum.asscalar() / n

def train1(net, train_images, train_labels, test_images, test_labels, loss, num_epochs, batch_size,
              params=None, lr=None, trainer=None):
    """Train and evaluate a model with CPU."""
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in data_iter(batch_size, train_images, train_labels):
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy1(test_images, test_labels, batch_size, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

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
    indices = nd.array(indices)
    nd.random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield nd.take(features, j, axis=0), nd.take(labels, j, axis=0)

(train_images, train_labels), (test_images, test_labels) = load_data()

batch_size = 256

net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

loss = gloss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

num_epochs = 20

#(256, 1, 28, 28)
#(256,)
train_images = nd.array(train_images).reshape((-1, 1, 28, 28)) / 255
train_labels = nd.array(train_labels)
test_images = nd.array(test_images).reshape((-1, 1, 28, 28)) / 255
test_labels = nd.array(test_labels)
train1(net, train_images, train_labels, test_images, test_labels, loss, num_epochs, batch_size, None, None, trainer)
