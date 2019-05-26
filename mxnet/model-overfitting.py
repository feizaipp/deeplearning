#! /usr/bin/python
#coding=utf-8

from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np
import matplotlib.pyplot as plt
import d2lzh as d2l

from pylab import *

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
    #plt.show()
    plt.savefig("./model-overfitting-s.png")

num_epochs, loss = 100, gloss.L2Loss()

def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    batch_size = min(10, train_labels.shape[0])
    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.01})
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(loss(net(train_features), train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features), test_labels).mean().asscalar())
    print('final epoch:train loss', train_ls[-1], 'test loss', test_ls[-1])
    print('weight:', net[0].weight.data().asnumpy(), '\nbias:', net[0].bias.data().asnumpy())
    my_plot(range(1, num_epochs + 1), train_ls, 'epoch', 'loss', range(1, num_epochs + 1), test_ls, ['train', 'test'])


# 生成数据集
# y=1.2x − 3.4x^2 + 5.6x^3 + 5 + ϵ
# ϵ 为噪声，服从均值为 0 ，标准差为 0.1 的正态分布
# 训练数据集和测试数据集样本都设为 100 
n_train, n_test, true_w, true_b = 100, 100, [1.2, 3.4, 5.6], 5
features = nd.random.normal(shape=(n_train + n_test, 1))
poly_features = nd.concat(features, nd.power(features, 2), nd.power(features, 3))

labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1] +
            true_w[2] * poly_features[:, 2] + true_b)

labels += nd.random.normal(scale=0.1, shape=labels.shape)

# 当样本数量不足时，出现过拟合现象
fit_and_plot(poly_features[0:2, :], poly_features[n_train:, :], labels[0:2], labels[n_train:])