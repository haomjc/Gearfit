# -*- coding: utf-8 -*-

# 把ResNet18加入程序

from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from Gear_data import gear_data

from visdom import Visdom

from Net import poly_model  # 导入神经网络模型
from ResNet import ResNet18


def get_batch(epoch):
    i = epoch % 15

    G, F = gear_data()

    x = [G[i]] * (
            1 + torch.randn(1, 6).cpu().numpy() * 0.01)  # 采用高斯随机分布增加样本数量

    # x=torch.tensor(x, dtype=torch.float).unsqueeze(1)
    x = torch.tensor(x, dtype=torch.float)
    # x = make_features(x)

    y = [F[i]]  # 变量y样本
    # y=torch.tensor(y, dtype=torch.float).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.float)
    # y = make_features(y)
    # print(y)

    return Variable(x), Variable(y)


# viz = Visdom(use_incoming_socket=False)
viz = Visdom()

viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))

# model = poly_model()          #选择神经网络模型

model = ResNet18()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-6, momentum=0.5)  # 加动量后收敛反而慢？
# optimizer = optim.SGD(model.parameters(), lr = 1e-5)

epoch = 1
print_loss = 0
while True:  # 定义loss到多少停止运算

    batch_x, batch_y = get_batch(epoch)
    # print(batch_x)
    output = model(batch_x)
    # print(output)
    loss = criterion(output, batch_y)
    # loss = criterion(output,batch_y)
    print_loss = print_loss + loss.item()
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    epoch += 1

    if epoch % 1000 == 0:
        viz.line([print_loss / 1000], [epoch], win='train_loss', update='append')
        print("epoch:", epoch)
        print("loss:%.10f\n" % (print_loss / 1000))

        if print_loss / 1000 < 1:  # 允许的MSELoss平均误差
            break
        print_loss = 0

torch.save(model, 'model.pkl')
