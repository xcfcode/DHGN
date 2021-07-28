# -*- coding: utf-8 -*-
# @Time    : 2020/3/17 8:31 下午
# @Author  : Xiachong Feng
# @File    : graph_encoder.py
# @Software: PyCharm
import torch
from torch import nn
from onmt.modules.graph_conv import GraphConv


class GraphEncoder(nn.Module):

    def __init__(self, node_encoder_out_dim, graph_encoder_in_dim, num_types, num_relations, n_heads, n_layers,
                 dropout=0.2, conv_name='hgt'):
        super(GraphEncoder, self).__init__()
        self.gnn_layers = nn.ModuleList()  # gnn layers
        self.num_types = num_types  # node type nums
        self.node_encoder_out_dim = node_encoder_out_dim  # node_encoder_out_dim
        self.graph_encoder_in_dim = graph_encoder_in_dim  # graph_encoder_in_dim
        self.adapts = nn.ModuleList()  # used to transform each type of node in to type-specific space
        self.dropout = nn.Dropout(dropout)  # dropout rate
        for t in range(num_types):
            # for every node type
            self.adapts.append(nn.Linear(node_encoder_out_dim, graph_encoder_in_dim))
        for layer in range(n_layers):
            self.gnn_layers.append(
                GraphConv(conv_name, graph_encoder_in_dim, graph_encoder_in_dim, num_types, num_relations, n_heads,
                          dropout))

    @classmethod
    def from_opt(cls, opt):
        return cls(opt.rnn_size, opt.graph_in_dim, opt.num_node_types, opt.num_edge_types, opt.graph_heads,
                   opt.graph_layers, dropout=opt.graph_dropout, conv_name=opt.graph_encoder)

    def forward(self, node_feature, node_type, edge_index, edge_type, node_position):
        """
        Graph Encoder Forward Pass
        :param node_feature: [num_nodes, node_encoder_out_dim]
        :param node_type: [num_nodes]
        :param edge_index: [2, num_edges]
        :param edge_type: [num_edges]
        :return:
        """
        res = torch.zeros(node_feature.size(0), self.graph_encoder_in_dim).to(
            node_feature.device)  # [num_nodes,hidden_dim]

        """first transform different type of nodes into different representation space"""
        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))  # index of t_id nodes
            if idx.sum() == 0:
                continue
            res[idx] = torch.tanh(self.adapts[t_id](node_feature[idx]))
        meta_xs = self.dropout(res)  # dropout [num_nodes, n_hid]
        del res

        """start to propagate message in graph encoder"""
        for gnn_layer in self.gnn_layers:
            meta_xs = gnn_layer(meta_xs, node_type, edge_index, edge_type, node_position)
        return meta_xs
