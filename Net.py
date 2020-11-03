import torch
import torch.nn as nn
import torch.nn.functional as F


class poly_model(nn.Module):
    def __init__(self):
        super(poly_model, self).__init__()
        self.poly1 = nn.Linear(4, 8)

        self.poly2 = nn.Linear(8, 32)

        self.poly3 = nn.Linear(32, 64)

        self.poly4 = nn.Linear(64, 128)

        self.poly5 = nn.Linear(128, 64)

        self.poly6 = nn.Linear(64, 15)
        self.dropout = nn.Dropout(p=0.5)  # 加一个dropout,防止过拟合

    def forward(self, x):
        out = self.poly1(x)
        out = F.leaky_relu(out)

        out = self.poly2(out)
        out = F.leaky_relu(out)

        out = self.poly3(out)
        out = F.leaky_relu(out)

        out = self.poly4(out)
        out = F.leaky_relu(out)

        out = self.poly5(out)
        out = F.leaky_relu(out)

        out = self.poly6(out)
        return out
