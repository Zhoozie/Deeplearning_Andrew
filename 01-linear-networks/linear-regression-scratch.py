"""
------------------------------------------------------------------------
从零开始实现线性回归
------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import torch
import random
from d2l import torch as d2l
 
def synthetic_data(w,b,num_examples):
    """
    构建带噪声的人造数据集
    param w: 权重
    param b: 偏差
    param num_examples: 样本数目
    """
    # 生成 X 均值为 0 ，方差为 1 ，num_examples个样本，列数为 w 的长度
    X = torch.normal(0,1,(num_examples,len(w)))
    # y = Xw+b
    y = torch.matmul(X,w)+b
    # 加入了一个均值为0，方差为0.01，形状同y相同的随机噪声
    y += torch.normal(0,0.01,y.shape)
    # 将X，y作为列向量返回
    return X,y.reshape((-1,1))
# 线性模型参数赋值
true_w = torch.tensor([2,-3.4])
true_b = 4.2
# features中的每一行都包含一个二维数据样本，labels中的每一行都包含一个一维标签值
features, labels = synthetic_data(true_w, true_b, 1000)
print('features:', features[0],'\nlabel:', labels[0])
# 画图看出特征和标签线性相关
d2l.set_figsize()
d2l.plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy(),1)
plt.show()

def data_iter(batch_size,features,labels):
    """
    生成小批量数据
    param batch_size: 批量大小
    param features: 特征矩阵
    param labels: 标签向量
    """
    num_examples = len(features)
    # 生成每个样本的index
    indices=list(range(num_examples))
    # 随机读取样本，无特定的顺序，随机打乱这些下标
    random.shuffle(indices)
    # 从 0 开始到 num_examples 结束每次跳 batch_size 的步长
    for i in range(0,num_examples,batch_size):
        # 从i开始，不超出预定的样本数
        batch_indices = torch.tensor(indices[i:min(i+batch_size,num_examples)])
        # 通过下标返回随机顺序的特征和随机顺序的标签
        yield  features[batch_indices],labels[batch_indices]
# 批量大小
batch_size = 10
# 读取第一个小批量数据样本并打印。 每个批量的特征维度显示批量大小和输入特征数。 同样的，批量的标签形状与batch_size相等。
for X,y in data_iter(batch_size,features, labels):
    print(X,'\n',y)
    break

# 定义初始化模型参数，均需要计算梯度
w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

def linreg(X,w,b):
    """
    定义模型：线性回归模型
    param X: 输入
    param w: 权重
    param b: 偏差
    y = Xw + b
    """
    return torch.matmul(X,w)+b

def squared_loss(y_hat,y):
    """
    定义损失函数：均方误差
    param y_hat: 预测值
    param y: 真实值
    """
    return(y_hat-y.reshape(y_hat.shape))**2/2

def sgd(params,lr,batch_size):
    """
    定义优化算法：小批量随机梯度下降法
    param params：包含w和b
    param lr: 学习率
    batch_size：批量大小，用于调整梯度以适配批量大小
    """
    # 不需要计算梯度
    with torch.no_grad():
        # 更新参数
        for param in params:
            # 求均值
            param -= lr*param.grad/batch_size
            # 将梯度设为0，为了下次计算与其不相关
            param.grad.zero_()

# 学习率
lr = 0.03
# 迭代次数
num_epochs = 3
# 模型
net = linreg
# 损失函数
loss = squared_loss
# 训练过程
for epoch in range(num_epochs):
    # 每次拿出一个批量大小的Xy
    for X,y in data_iter(batch_size, features, labels):
        # X,y的小批量损失,放进net作损失，预测的y和真实的y做损失
        l = loss(net(X,w,b),y)
        # 求和之后反向传播 
        l.sum().backward()
        # 使用梯度更新w,b
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        # 计算预测和真实标签的损失
        train_l = loss(net(features,w,b),labels)
        print(f'epoch{epoch+1},loss{float(train_l.mean()):f}')

# 比较真实参数和通过训练得到的参数来评估训练的成功程度
print(f'w的估计误差：{true_w-w.reshape(true_w.shape)}')
print(f'b的估计误差：{true_b-b}')

