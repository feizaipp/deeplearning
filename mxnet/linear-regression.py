#! /usr/bin/python
#coding=utf-8

from mxnet import autograd, nd
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet import init
from mxnet.gluon import loss as gloss
from mxnet import gluon

num_imputs = 2
num_examples = 1000
true_w = [5.6, -8]
true_b = 2.5
featrues = nd.random.normal(scale=1, shape=(num_examples, num_imputs))
lables = true_w[0] * featrues[:, 0] + true_w[1] * featrues[:, 1] + true_b
lables += nd.random.normal(scale=0.1, shape=lables.shape)

batch_size = 10
datasize =  gdata.ArrayDataset(featrues, lables)
# shuffle=True 随机读取
data_iter = gdata.DataLoader(datasize, batch_size, shuffle=True)

# 定义模型
net = nn.Sequential()
# Dense:全连接层
# 1:该层输出个数
# 在 Gluon 中我们无须指定每一层输入的形状，例如线性回归的输入个数。
# 当模型得到数据时，例如后面执行 net(X) 时，模型将自动推断出每一层的输入个数。
net.add(nn.Dense(1))

# 初始化模型参数
net.initialize(init.Normal(sigma=0.01))

# 定义损失函数
# 平方损失又称L2范数损失
loss = gloss.L2Loss()

# 定义优化算法
# 小批量随机梯度下降
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate' : 0.03})

# 训练模型
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        # 为了减少计算和内存开销，默认条件下 MXNet 不会记录用于求梯度的计算。
        # 我们需要调用 record 函数来要求 MXNet 记录与求梯度有关的计算。
        with autograd.record():
            l = loss(net(X), y)
        # 对模型参数求梯度
        l.backward()
        # step函数来迭代模型参数
        trainer.step(batch_size)
    # 计算当前训练得到的参数对应的损失
    l = loss(net(featrues), lables)
    print('epoch %d, loss:%f' % (epoch, l.mean().asnumpy()))

dense = net[0]
print(true_w)
print(dense.weight.data())
print(true_b)
print(dense.bias.data())