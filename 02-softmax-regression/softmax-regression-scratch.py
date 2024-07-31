"""
------------------------------------------------------------------------
从零开始实现Softmax
------------------------------------------------------------------------
"""
from IPython import display
from mxnet import autograd, gluon, np, npx
import torch
from d2l import mxnet as d2l

# 设置批量大小
batch_size = 256
# 引入Fashion-MNIST数据集
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 初始化模型参数
num_inputs = 784
num_outputs = 10
# 创建权重矩阵 W 和偏置向量 b  
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
X.sum(0, keepdim=True), X.sum(1, keepdim=True)
def softmax(X):
    """
    定义softmax操作
    :param X: 一个二维张量
    :return: 每个样本在每个类别上的概率分布。
    """
    # 计算 X 中每个元素的指数
    X_exp = torch.exp(X)
    # 沿着第一个维度（即每个样本的维度）对 X_exp 进行求和，keepdim=True 保留了输出的维度，
    partition = X_exp.sum(1, keepdim=True)
    # 利用了广播机制
    return X_exp / partition

X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, X_prob.sum(1)

def net(X):
    """
    定义softmax模型
    :param X: 一个二维张量
    :return: softmax输出  
    """
    # X.reshape((-1, W.shape[0]))中的-1表示自动推断的维度大小，
    # 整体表示将X变成一个N行W列的张量，其中N表示批次大小，W表示权重矩阵的列数
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# y_hat[[0, 1], y]表示从y_hat张量中选择指定索引位置的元素.
y_hat[[0, 1], y]
def cross_entropy(y_hat, y):
    """
    定义交叉熵函数
    :param y_hat: softmax层的输出
    :param y: 真实标签的索引
    :return: 整个批次的平均交叉熵损失  
    """
    return - torch.log(y_hat[range(len(y_hat)), y])
cross_entropy(y_hat, y)

def accuracy(y_hat, y): 
    """
    计算预测正确的数量
    :param y_hat: 模型预测的输出，如果是softmax输出，则需要转换为类别索引  
    :param y: 真实标签  
    :return: 计算预测正确的数量
    """
    # 如果y_hat是softmax输出，则转换为类别索引
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    # 比较预测值和真实值 
    cmp = y_hat.type(y.dtype) == y
    # 计算正确的样本数
    return float(cmp.type(y.dtype).sum())

res = accuracy(y_hat, y) / len(y)
print(res)

class Accumulator:
    """
    在n个变量上累加
    """
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
        """zip(self.data, args)将self.data列表和传入的参数一一对应进行打包，生成一个迭代器
        在每次迭代中，a表示self.data中的元素，b表示传入的参数中的元素"""

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net, data_iter):
    """
    计算在指定数据集上模型的精度
    """
    # 如果是使用torch.nn实现的模型
    if isinstance(net, torch.nn.Module):
        # 将模型设置为评估模式
        net.eval()
    # 正确预测数、预测总数
    metric = Accumulator(2)
    # 创建累加器，保存累计值-正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    # 分类正确样本数除总样本数
    return metric[0] / metric[1]
print(evaluate_accuracy(net, test_iter))

def train_epoch_ch3(net, train_iter, loss, updater):
    """
    训练模型一个迭代周期
    """
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

class Animator:
    """
    在动画中绘制数据
    """
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1: # 判断是否只有1个子图
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数，直接调用d2l.set_axes函数配置子图属性
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"): 
            y = [y]
        # 判断y是否是可迭代的对象，若不是，则将其转换为一个只包含单个元素的列表
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        # 判断参数x是否是可迭代的对象，如果不是，则将其转换为一个只包含x值的列表，并重复n次，以匹配y的长度
        if not self.X:
            self.X = [[] for _ in range(n)]
        # 如果self.X为空，创建一个包含n个空列表的列表，用于存储每个数据点的x值，下同
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
            # 遍历参数x和y的元素对，将非None的值分别添加到对应的self.X和self.Y列表中的对应位置
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
        # 清除输出区域的内容，并设置等待下一次输出

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """
    训练模型
    """
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
    """使用assert语句进行断言检查，确保训练损失、训练准确率和测试准确率满足一些条件，
    否则将会引发AssertionError异常，提示训练过程中出现了异常情况。
    """

lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

def predict_ch3(net, test_iter, n=6):
    """
    预测标签
    """
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    d2l.plt.show()

predict_ch3(net, test_iter)