import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv
from dgl import function as fn

from agg import *

class NAGNN(nn.Module):
    def __init__(self, g, param):
        super(NAGNN, self).__init__()

        self.relu = nn.ReLU()

        self.g = g
        self.param = param
        self.in_dim = param['in_dim']  # 输入特征维度
        self.hidden_dim = param['hidden_dim']  # 隐藏层维度
        self.graph_dim = param['graph_dim']
        self.out_dim = param['out_dim']  # 输出维度（最终预测类别数）
        self.dataset = param['dataset']
        self.dropout = param['dropout']
        self.k = param['k']

        self.w0 = nn.Parameter(torch.randn(self.in_dim, self.hidden_dim) * 0.01)
        self.a = nn.Parameter(torch.randn(2 * self.hidden_dim))
        self.beta = param['self_nei']
        self.tau=param['tau']



        self.normalization1 = nn.LayerNorm(self.in_dim)

        self.lin0 = nn.Linear(self.in_dim, self.hidden_dim)
        if self.k >1 :
            for i in range(2,self.k):
                self.lin_i = nn.Linear(3 * self.hidden_dim, self.hidden_dim)
            self.lin_f = nn.Linear(3 * self.hidden_dim, self.out_dim)
        elif self.k == 1:
            self.lin_f = nn.Linear(3 * self.hidden_dim, self.out_dim)
        elif self.k==0:
            self.lin_f = nn.Linear(self.hidden_dim, self.out_dim)
        elif self.k < 0 :
            raise ValueError(f"k should be no less than 0")


        self.drop = nn.Dropout(self.dropout)

        #self.reset_parameters()

        nn.init.kaiming_uniform_(self.w0)
        nn.init.kaiming_uniform_(self.lin0.weight)

        for i in range(2,self.k):
            nn.init.kaiming_uniform_(self.lin_i.weight)
        nn.init.kaiming_uniform_(self.lin_f.weight)

    def reset_parameters(self):
        """重新初始化所有可训练参数"""
        nn.init.kaiming_uniform_(self.w0)
        nn.init.kaiming_uniform_(self.lin0.weight)
        if self.k>1:
            for i in range(2,self.k):
                nn.init.kaiming_uniform_(self.lin_i.weight)
        nn.init.kaiming_uniform_(self.lin_f.weight)


    def forward(self, features):
        """
        features: input node features (shape: [num_nodes, in_dim])
        g: graph object, passed into __init__
        """

        features = self.normalization1(features)
        g = dgl.add_self_loop(self.g)
        #g = self.g

        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)  # D^(-1/2)
        norm[torch.isinf(norm)] = 0
        g.ndata['h'] = features
        g.ndata['norm'] = norm.unsqueeze(1)
        g.apply_edges(lambda edges: {'norm': edges.src['norm'] * edges.dst['norm']})
        g.update_all(fn.u_mul_e('h', 'norm', 'm'), fn.sum('m', 'h_w'))
        X_norm = g.ndata.pop('h_w')

        h_l0 = self.lin0(features)
        h_c0 = torch.mm(X_norm,self.w0)

        h0 = F.relu( self.beta * h_c0 + (1-self.beta) * h_l0)

        if self.k>0:
            h_k = compute_attention_scores_and_aggregate(self.g, self.a, self.tau, h0, h0)
            #self.g.ndata['h_temp'] = h0
            #self.g.update_all(fn.copy_u('h_temp', 'm'), fn.mean('m', 'h_neigh'))
            #h_neigh = self.g.ndata.pop('h_neigh')

            h_k = torch.cat([h0, h_k], dim=1)

            
            h_k = self.drop(h_k)
            if self.k>1:
               for i in range(2,self.k):
                   h_k = self.relu(self.lin_i(h_k))
                   h_k = compute_attention_scores_and_aggregate(self.g, self.a, self.tau, h0, h_k)
                   h_k = torch.cat([h_k, h_neigh], dim=1)


                   h_k = self.drop(h_k)
        elif self.k==0:
            h_k = self.drop(h0)

        h_f = self.lin_f(h_k)

        out = h_f

        return out
