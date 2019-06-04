import d2lzh as d2l
from mxnet import autograd, nd
from mxnet.gluon import nn

# 互相关运算
def corr2d(X, K):
    h, w = K.shape
    Y = nd.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = nd.array([[0, 1], [2, 3]])
out = corr2d(X, K)
print(out)

# 物体边缘检测
X = nd.ones((6, 8))
X[:, 2:6] = 0
print(X)
# 从白到黑的边缘和从黑到白的边缘分别检测成了1和-1
K = nd.array([[1, -1]])
Y = corr2d(X, K)
print(Y)

# 通过数据学习核数组
# 构造一个输出通道数为1（将在“多输入通道和多输出通道”一节介绍通道），核数组形状是(1, 2)的二
# 维卷积层
conv2d = nn.Conv2D(1, kernel_size=(1, 2))
conv2d.initialize()

# 二维卷积层使用4维输入输出，格式为(样本, 通道, 高, 宽)，这里批量大小（批量中的样本数）和通
# 道数均为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
    l.backward()
    # 简单起见，这里忽略了偏差
    conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
    if (i + 1) % 2 == 0:
        print('batch %d, loss %.3f' % (i + 1, l.sum().asscalar()))

K = conv2d.weight.data().reshape((1, 2))
print(K)

# 填充
# (𝑛ℎ−𝑘ℎ+𝑝ℎ+1)×(𝑛𝑤−𝑘𝑤+𝑝𝑤+1)
# 定义一个函数来计算卷积层。它初始化卷积层权重，并对输入和输出做相应的升维和降维
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # (1, 1)代表批量大小和通道数（“多输入通道和多输出通道”一节将介绍）均为1
    X = X.reshape((1, 1) + X.shape)
    print(X.shape)
    Y = conv2d(X)
    print(Y.shape)
    return Y.reshape(Y.shape[2:])  # 排除不关心的前两维：批量和通道

# 注意这里是两侧分别填充1行或列，所以在两侧一共填充2行或列
# 维度： (8-3+1*2+1) * (8-3+1*2+1)
conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = nd.random.uniform(shape=(8, 8))
print(comp_conv2d(conv2d, X).shape)

# 使用高为5、宽为3的卷积核。在高和宽两侧的填充数分别为2和1
# # 维度： (8-5+2*2+1) * (8-3+1*2+1)
conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape)

# 步幅
# ⌊(𝑛ℎ−𝑘ℎ+𝑝ℎ+𝑠ℎ)/𝑠ℎ⌋×⌊(𝑛𝑤−𝑘𝑤+𝑝𝑤+𝑠𝑤)/𝑠𝑤⌋
# 下面我们令高和宽上的步幅均为2，从而使输入的高和宽减半
# 维度： (8-3+1*2+2)/2 * (8-3+1*2+2)/2
conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
print(comp_conv2d(conv2d, X).shape)

# 维度： (8-3+0*2+3)/3 * (8-5+1*2+4)/4
conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
print(comp_conv2d(conv2d, X).shape)

# 池化层
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = nd.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
Y = pool2d(X, (2, 2))
print(Y)
Y = pool2d(X, (2, 2), 'avg')
print(Y)

# 多输入通道和多输出通道
# 多输入通道
def corr2d_multi_in(X, K):
    # 首先沿着X和K的第0维（通道维）遍历。然后使用*将结果列表变成add_n函数的位置参数
    # （positional argument）来进行相加
    return nd.add_n(*[d2l.corr2d(x, k) for x, k in zip(X, K)])

X = nd.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
              [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = nd.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

Y = corr2d_multi_in(X, K)
print(Y)

# 多输出通道
def corr2d_multi_in_out(X, K):
    # 对K的第0维遍历，每次同输入X做互相关计算。所有结果使用stack函数合并在一起
    return nd.stack(*[corr2d_multi_in(X, k) for k in K])

K = nd.stack(K, K + 1, K + 2)
Y = corr2d_multi_in_out(X, K)
print(Y)

# 1×1 卷积层
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = nd.dot(K, X)  # 全连接层的矩阵乘法
    return Y.reshape((c_o, h, w))

X = nd.random.uniform(shape=(3, 3, 3))
K = nd.random.uniform(shape=(2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
print(Y1)
print(Y2)
print((Y1 - Y2).norm().asscalar() < 1e-6)

# 池化层中的填充和步幅
X = nd.arange(16).reshape((1, 1, 4, 4))
pool2d = nn.MaxPool2D(3)
Y = pool2d(X)
print(Y)

pool2d = nn.MaxPool2D(3, padding=1, strides=2)
Y = pool2d(X)
print(Y)

pool2d = nn.MaxPool2D((2, 3), padding=(1, 2), strides=(2, 3))
Y = pool2d(X)
print(Y)

# 池化层中的多通道
# 在处理多通道输入数据时，池化层对每个输入通道分别池化，而不是像卷积层那样将各通道的输入按通道相加。
X = nd.concat(X, X + 1, dim=1)
print(X)

pool2d = nn.MaxPool2D(3, padding=1, strides=2)
Y = pool2d(X)
print(Y)
