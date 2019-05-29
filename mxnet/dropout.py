#! /usr/bin/python
#coding=utf-8

from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
import d2lzh as d2l

def dropout(X, drop_prob):
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob

    if keep_prob == 0:
        return X.zeros_like()

    mask = nd.random.uniform(0, 1, X.shape)
    mask =  mask < keep_prob
    return mask * X / keep_prob

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
drop_prob1, drop_prob2 = 0.2, 0.5
num_epochs, lr, batch_size = 5, 0.5, 256
loss = gloss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential()
net.add(nn.Dense(num_hiddens1, activation="relu"),
            nn.Dropout(drop_prob1),
            nn.Dense(num_hiddens2, activation="relu"),
            nn.Dropout(drop_prob2),
            nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate' : lr})
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)

