"""
------------------------------------------------------------------------
线性回归的简单实现
------------------------------------------------------------------------
"""
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
 
# 构造真实的w和b，然后通过人工数据合成函数生成我们需要的features和labels
true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels = d2l.synthetic_data(true_w,true_b,1000)
 
# 将features和labels作为API的参数传递，并通过数据迭代器指定batch_size。布尔值is_train表示是否希望数据迭代器对象在每个迭代周期内打乱数据。
def load_array(data_arrays,batch_size,is_train=True):
    """
    构造一个pytorch数据迭代器
    param data_arrays: features和labels
    param batch_size: 批量大小
    param is_train: 控制是否希望数据迭代器对象在每个迭代周期内打乱数据
    """
    dataset = data.TensorDataset(*data_arrays)
    # 使用 data.DataLoader 负责批量地、可选地打乱（如果 is_train=True）并加载数据集
    return data.DataLoader(dataset,batch_size,shuffle=is_train)
 
batch_size = 10
data_iter = load_array((features,labels),batch_size)
 
#为了验证是否正常工作，让我们读取并打印第一个小批量样本。 与 3.2节不同，这里我们使用iter构造Python迭代器，并使用next从迭代器中获取第一项。
print(next(iter(data_iter)))
 
from torch import nn
# 定义一个模型变量net，它是一个Sequential类的实例。 Sequential类将多个层串联在一起。 当给定输入数据时，Sequential实例将数据传入到第一层， 然后将第一层的输出作为第二层的输入，以此类推。
# nn.Linear有两个参数：第一个指定输入特征形状，第二个指定输出特征形状。
net = nn.Sequential(nn.Linear(2,1))

# 将第一个线性层的权重参数（weight）的初始值设置为均值为0，标准差为0.01的正态分布随机值。
net[0].weight.data.normal_(0,0.01)
# 将第一个线性层的偏置参数（bias）的初始值全部设置为0
net[0].bias.data.fill_(0)
 
# 使用MSELoss类定义均方误差
loss = nn.MSELoss()

# 实例化SGD实例，
# 第一个参数是获取网络中的所有参数；第二个参数是学习率
trainer = torch.optim.SGD(net.parameters(),lr=0.03)
 
# 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X,y in data_iter:
        # net（）这里本身带了模型参数，不需要把w和b放进去了，net(X)是预测值，y是真实值，拿到预测值和真实值做Loss
        I = loss(net(X),y)
        # 梯度清零
        trainer.zero_grad()
        # 计算反向传播，这里pytorch已经做了sum就不需要在做sum了（loss是一个张量，求sum之后是标量）
        I.backward()
        # 有了梯度之后调用step（）函数来进行一次模型的更新。调用step函数，从而分别更新权重和偏差
        trainer.step()
    # 当扫完一遍数据之后，把所有的feature放进network中，和所有的Label作一次Loss
    I = loss(net(features),labels)
    print(f'epoch{epoch+1},loss{I:f}')