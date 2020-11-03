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
    i = epoch % 8

    G0, F0, G1, F1, G2, F2, G3, F3, G4, F4, G5, F5, G6, F6, G7, F7, G8, F8, G9, F9, G10, F10, G11, F11, G12, F12, G13, F13, G14, F14, G15, F15, G16, F16, G17, F17 = gear_data()

    x = [G0, G1, G2, G3, G4, G5, G6, G7, G8, G9, G10, G11, G12, G13, G14, G15, G16, G17] * (
            1 + torch.randn(1, 6).cpu().numpy() * 0.01)  # 采用高斯随机分布增加样本数量

    # x=torch.tensor(x, dtype=torch.float).unsqueeze(1)
    x = torch.tensor(x, dtype=torch.float)
    # x = make_features(x)

    y = [F0, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17]  # 变量y样本
    # y=torch.tensor(y, dtype=torch.float).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.float)
    # y = make_features(y)
    # print(y)

    return Variable(x), Variable(y)


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
