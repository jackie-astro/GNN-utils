import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # W 是论文中的W [1433x8]，将原始[2708,1433]的节点特征矩阵降维成[2708,8]
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # a 是论文中的a [16, 1]，将两个拼接后的节点[2708,8*2]的节点特征矩阵降维成[2708]
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)  # 使用W对节点特征矩阵进行降维：2708x1433, 1433x8 --》2708x8
        N = h.size()[0]

        # 拼接一个大矩阵，用来计算注意力系数矩阵 ： [2708x2708x16]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)

        # 计算注意力系数矩阵：[2708x2708x16] x [16x1] --> [2708x2708]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # softmax，让注意力系数和为1
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 更新节点特征矩阵 [2708x2708] x [2708x8] ---> [2708x8]
        h_prime = torch.matmul(attention, h) 

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime   # 2708x8

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


