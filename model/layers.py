import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spmm
from sparsemax import Sparsemax


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha, inplace=True)
        self.sparsemax = Sparsemax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = torch.zeros_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        del zero_vec
        attention = self.dropout(self.sparsemax(attention)) # attention.shape = d x u x t x t
        
        h_prime = torch.matmul(attention, Wh) # h_prime.shape = d x u x t x out_features
        del Wh
        del attention

        h_prime = self.dropout(h_prime)

        if self.concat:
            return F.elu(h_prime, inplace=True)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(-2, -1) # e.shape = d x u x t x t
        del Wh1
        del Wh2
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.W = nn.Linear(in_features, out_features)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha, inplace=True)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]

        edge = adj.t()
        del adj

        # h = torch.mm(input, self.W)
        h = self.W(input)
        del input
        # h: N x out

        edge_h1 = torch.matmul(self.a[:, :self.out_features], h.t()).squeeze()
        edge_h2 = torch.matmul(self.a[:, self.out_features:], h.t()).squeeze()
        # edge_h1: N, edge_h2: N

        edge_e = torch.exp(self.leakyrelu(edge_h1[edge[0, :]] + edge_h2[edge[1, :]]))
        assert not torch.isnan(edge_e).any()
        # edge_e: E
        del edge_h1
        del edge_h2

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        del edge_e
        # h_prime: N x out
        del edge
        del h

        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()
        del e_rowsum
        h_prime = self.dropout(h_prime)

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime, inplace=True)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'