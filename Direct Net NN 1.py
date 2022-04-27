import torch
import numpy as np
import torch.nn as nn

epoch_max = 64
sample_batch = 64
sample_pos = 3
sample_prope = 3
goal_batch = 2
dorate = 0.000025
m = nn.Sigmoid()

s = torch.randn(sample_batch, sample_pos, sample_prope)     # 超越2维的矩阵会以后两个维合成一个（类二维）元组，之前的所有维度为Batch维度
y = torch.randn(sample_batch, sample_pos, goal_batch)
y_s = m(y)

hid1 = 16
w1 = torch.randn(sample_batch, sample_pos, hid1)

hid2 = 8
w2 = torch.randn(sample_batch, hid1, hid2)

hid3 = 4
w3 = torch.randn(sample_batch, hid2, hid3)

hid4 = goal_batch
w4 = torch.randn(sample_batch, hid3, hid4)

for epoch in range(0, epoch_max):

    h1 = s.bmm(w1)
    h1 = h1.clamp(min=0)

    h2 = h1.bmm(w2)
    h2 = h2.clamp(min=0)

    h3 = h2.bmm(w3)
    h3 = h3.clamp(min=0)

    h4 = h3.bmm(w4)
    Y = h4.clamp(min=0)

    Y_s = m(Y)

    Y_loss = (Y_s - y_s).pow(2).sum()
    Y_loss = np.around(Y_loss, decimals=6)
    print("Eopch:{}, Y_Loss:{:.6f}".format(epoch + 1, Y_loss))

    grad_Y = 2 * (Y - y)

    #求导

    grad_w4 = h3.transpose(1, 2).bmm(grad_Y)
    grad_h3 = grad_Y.bmm(w4.transpose(1, 2))

    grad_w3 = h2.transpose(1, 2).bmm(grad_h3)
    grad_h2 = grad_h3.bmm(w3.transpose(1, 2))

    grad_w2 = h1.transpose(1, 2).bmm(grad_h2)
    grad_h1 = grad_h2.bmm(w2.transpose(1, 2))

    grad_w1 = s.transpose(1, 2).bmm(grad_h1)

    w1 = w1 - dorate * grad_w1
    w2 = w2 - dorate * grad_w2
    w3 = w3 - dorate * grad_w3
    w4 = w4 - dorate * grad_w4

print("w1", w1)
print("w2", w2)
print("w3", w3)
print("w4", w4)

