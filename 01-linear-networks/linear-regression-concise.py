"""
    线性回归的简单实现
"""
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
 
#1.生成数据集
#构造真实的w和b，然后通过人工数据合成函数生成我们需要的features和labels
true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels = d2l.synthetic_data(true_w,true_b,1000)
 
#2.读取数据集
#将features和labels作为API的参数传递，并通过数据迭代器指定batch_size。布尔值is_train表示是否希望数据迭代器对象在每个迭代周期内打乱数据。
def load_array(data_arrays,batch_size,is_train=True):
    """构造一个pytorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)
 
batch_size = 10
data_iter = load_array((features,labels),batch_size)
 
#为了验证是否正常工作，让我们读取并打印第一个小批量样本。 与 3.2节不同，这里我们使用iter构造Python迭代器，并使用next从迭代器中获取第一项。
print(next(iter(data_iter)))
 
#3.定义模型
#nn是神经网络的缩写
from torch import nn
#定义一个模型变量net，它是一个Sequential类的实例。 Sequential类将多个层串联在一起。 当给定输入数据时，Sequential实例将数据传入到第一层， 然后将第一层的输出作为第二层的输入，以此类推。
#在下面的例子中，我们的模型只包含一个层，因此实际上不需要Sequential。 但是由于以后几乎所有的模型都是多层的，在这里使用Sequential会让你熟悉“标准的流水线”。
#这一单层被称为全连接层,全连接层在Linear类中定义
#在构造nn.Linear时指定输入和输出尺寸:将两个参数传递到nn.Linear中。 第一个指定输入特征形状，即2，第二个指定输出特征形状，输出特征形状为单个标量，因此为1
net = nn.Sequential(nn.Linear(2,1))
 
#4.初始化模型参数
#在使用net之前，我们需要初始化模型参数.深度学习框架通常有预定义的方法来初始化参数
#指定每个权重参数应该从均值为0、标准差为0.01的正态分布中随机采样， 偏置参数将初始化为零。
#同构造nn.Linear时指定输入和输出尺寸，能直接访问参数以设定它们的初始值。
# 我们通过net[0]选择网络中的第一个图层， 然后使用weight.data和bias.data方法访问参数。 我们还可以使用替换方法normal_和fill_来重写参数值。
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)
 
#5.定义损失函数
#算均方误差使用的是MSELoss类，也称为平方范数。 默认情况下，它返回所有样本损失的平均值。
loss = nn.MSELoss()
 
#6.定义优化算法
#小批量随机梯度下降算法是一种优化神经网络的标准工具， PyTorch在optim模块中实现了该算法的许多变种。
#实例化一个SGD实例时，我们要指定优化的参数 （可通过net.parameters()从我们的模型中获得）以及优化算法所需的超参数字典。
# 小批量随机梯度下降只需要设置lr值，这里设置为0.03。
trainer = torch.optim.SGD(net.parameters(),lr=0.03)
 
#7.训练
num_epochs = 3
for epoch in range(num_epochs):
    for X,y in data_iter:
        I = loss(net(X),y) #net（）这里本身带了模型参数，不需要把w和b放进去了，net(X)是预测值，y是真实值，拿到预测值和真实值做Loss
        trainer.zero_grad() #梯度清零
        I.backward() #计算反向传播，这里pytorch已经做了sum就不需要在做sum了（loss是一个张量，求sum之后是标量）
        trainer.step() #有了梯度之后调用step（）函数来进行一次模型的更新。调用step函数，从而分别更新权重和偏差
    I = loss(net(features),labels)  #当扫完一遍数据之后，把所有的feature放进network中，和所有的Label作一次Loss
    print(f'epoch{epoch+1},loss{I:f}')