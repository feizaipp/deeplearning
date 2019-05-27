#! /usr/bin/python
#coding=utf-8

from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import d2lzh as d2l

# 为了较容易地观察过拟合，使用高维线性回归问题做试验，
# 如设维度p=200；同时，我们特意把训练数据集的样本数设低，如20。
n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = nd.ones((num_inputs, 1)) * 0.01, 0.05

features = nd.random.normal(shape=(n_train + n_test, num_inputs))
labels = nd.dot(features, true_w) + true_b
labels += nd.random.normal(scale=0.1, shape=labels.shape)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train, :], labels[n_train:]

batch_size, num_epochs, lr = 1, 100, 0.003
train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size=batch_size, shuffle=True)

def squared_loss(y_hat, y):
    """Squared loss."""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

loss = squared_loss

def my_plot(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(8, 6)):
    figure(figsize=figsize, dpi=80)
    plt.plot(x_vals, y_vals, color="red", linewidth=1)
    if x2_vals and y2_vals:
        plt.scatter(x2_vals, y2_vals, color="blue", s=10)
    plt.legend(legend)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tick_params(axis='x',which='major',labelsize=20)
    plt.tick_params(axis='y',which='major',labelsize=20)
    plt.axis([0, 100, 0.01, 100])
    plt.show()
    #plt.savefig("./L2-regularization2-s.png")

def fit_and_plot_gluon(wd):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=1))
    # 对权重参数衰减。权重名称以 weight 结尾
    trainer_w = gluon.Trainer(net.collect_params('.*weight'), 'sgd', {'learning_rate' : lr, 'wd' : wd})
    # 不对偏差参数衰减。
    trainer_b = gluon.Trainer(net.collect_params('.*bias'), 'sgd', {'learning_rate' : lr})
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            # 对里那个圪Trainer实例分别调用step函数，从而分别更新权重和偏差
            trainer_w.step(batch_size)
            trainer_b.step(batch_size)
        train_ls.append(loss(net(train_features), train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features), test_labels).mean().asscalar())
    my_plot(range(1, num_epochs + 1), train_ls, 'epoch', 'loss', range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print("L2 norm of w:", net[0].weight.data().norm().asscalar())

# 不使用权重衰减
#fit_and_plot_gluon(0)

# 使用权重衰减
fit_and_plot_gluon(3)