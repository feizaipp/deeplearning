#! /usr/bin/python
#coding=utf-8

from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

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


