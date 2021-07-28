# -*- coding: utf-8 -*-
# @Time    : 2020/3/22 3:45 下午
# @Author  : Xiachong Feng
# @File    : graph_conv.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv, RGCNConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
import math


class HGTConv(MessagePassing):
    """
    Heterogeneous Graph Transformer
    Note that original HGT aggregate information from all other nodes, but in our setting,
    the utterance node needs to get all speaker information and a distribution from knowledge nodes. (*****)
    Paper : https://arxiv.org/abs/2003.01332
    This Code is borrowed from : https://github.com/acbull/pyHGT

    """

    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout=0.2, **kwargs):
        super(HGTConv, self).__init__(aggr='add', **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types
        self.num_relations = num_relations
        self.total_rel = num_types * num_relations * num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()

        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))

        '''
            TODO: make relation_pri smaller, as not all <st, rt, tt> pair exist in meta relation list.
        '''
        self.relation_pri = nn.Parameter(torch.ones(num_types, num_relations, num_types, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))

        self.relation_s2u = nn.Parameter(torch.Tensor(1, out_dim, out_dim))

        self.skip = nn.Parameter(torch.ones(num_types))
        self.drop = nn.Dropout(dropout)

        glorot(self.relation_att)
        glorot(self.relation_msg)
        glorot(self.relation_s2u)

    def forward(self, node_inp, node_type, edge_index, edge_type, node_position):
        return self.propagate(edge_index, node_inp=node_inp, node_type=node_type, edge_type=edge_type,
                              node_position=node_position)

    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type, node_position_i,
                node_position_j):

        '''
            j: source, i: target; <j, i>
        '''
        data_size = edge_index_i.size(0)
        '''
            Create Attention and Message tensor beforehand.
        '''
        res_att = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)

        for source_type in range(self.num_types):
            sb = (node_type_j == int(source_type))
            k_linear = self.k_linears[source_type]
            v_linear = self.v_linears[source_type]
            for target_type in range(self.num_types):
                tb = (node_type_i == int(target_type)) & sb
                q_linear = self.q_linears[target_type]
                for relation_type in range(self.num_relations):
                    '''
                        idx is all the edges with meta relation <source_type, relation_type, target_type>
                    '''
                    idx = (edge_type == int(relation_type)) & tb
                    if idx.sum() == 0:
                        continue
                    '''
                        Get the corresponding input node representations by idx.
                        Add utterance position encoding to source representation (j)
                    '''
                    target_node_vec = node_inp_i[idx]
                    source_node_vec = node_inp_j[idx]

                    '''
                        Step 1: Heterogeneous Mutual Attention
                    '''
                    q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = torch.bmm(k_mat.transpose(1, 0), self.relation_att[relation_type]).transpose(1, 0)

                    res_att[idx] = (q_mat * k_mat).sum(dim=-1) * \
                                   self.relation_pri[target_type][relation_type][source_type] / self.sqrt_dk
                    '''
                        Step 2: Heterogeneous Message Passing
                    '''
                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    res_msg[idx] = torch.bmm(v_mat.transpose(1, 0), self.relation_msg[relation_type]).transpose(1, 0)
        '''
            Softmax based on target node's id (edge_index_i). Store attention value in self.att for later visualization.
        '''

        self.att = softmax(res_att, edge_index_i)

        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg
        return res.view(-1, self.out_dim)

    def update(self, aggr_out, node_inp, node_type, edge_index_i, edge_index_j, edge_type):
        """
        Step 3: Target-specific Aggregation
        x = W[node_type] * gelu(Agg(x)) + x
        :param aggr_out:
        :param node_inp:
        :param node_type:
        :return:
        """

        """Add speaker information for each utterance (post message passing)"""
        speaker_information = torch.zeros_like(aggr_out).to(node_inp.device)  # [node num, dim]

        utterance2speaker_egde = (edge_type == 0)

        utterance2speaker_index_j = edge_index_j[utterance2speaker_egde]  # source utterance
        utterance2speaker_index_i = edge_index_i[utterance2speaker_egde]  # target speaker

        # get speaker original rep
        speaker_input_rep = node_inp[utterance2speaker_index_i]
        # node type aware transformation speaker node type : 1
        speaker_input_rep = self.v_linears[1](speaker_input_rep)  # [speaker_num, dim] self.v_linears[1] for node type 1
        # relation aware message
        speaker_message = torch.bmm(speaker_input_rep.unsqueeze(0), self.relation_s2u).squeeze(0)  # [speaker_num, dim]
        speaker_information[utterance2speaker_index_j] = speaker_message
        # add speaker information to utterances
        aggr_out = aggr_out + speaker_information

        # activation function
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)

        # target specific aggregation
        for target_type in range(self.num_types):
            idx = (node_type == int(target_type))
            if idx.sum() == 0:
                continue
            '''
                Add skip connection with learnable weight self.skip[t_id]
            '''
            alpha = F.sigmoid(self.skip[target_type])
            res[idx] = self.a_linears[target_type](aggr_out[idx]) * alpha + node_inp[idx] * (1 - alpha)
        return self.drop(res)

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)


class GraphConv(nn.Module):
    def __init__(self, conv_name, in_hid, out_hid, num_types, num_relations, n_heads, dropout):
        """
        Different type of convolution operation
        :param conv_name: graph convolution operation [hgt, gcn, gat]
        :param in_hid: input dim
        :param out_hid: output dim
        :param num_types: node type nums
        :param num_relations: num relation types
        :param n_heads: self-attention heads
        :param dropout: dropout rate
        """
        super(GraphConv, self).__init__()
        self.conv_name = conv_name
        if self.conv_name == 'hgt':
            self.conv = HGTConv(in_hid, out_hid, num_types, num_relations, n_heads, dropout)
        elif self.conv_name == 'gcn':
            self.conv = GCNConv(in_hid, out_hid)
        elif self.conv_name == "rgcn":
            self.conv = RGCNConv(in_hid, out_hid, num_relations, num_bases=30)
        elif self.conv_name == 'gat':
            self.conv = GATConv(in_hid, out_hid // n_heads, heads=n_heads)

    def forward(self, meta_xs, node_type, edge_index, edge_type, node_position):
        """
        gnn layer forward pass
        :param meta_xs: [node_nums, graph_encoder_in_dim]
        :param node_type: [node_nums]
        :param edge_index: [2,num edges] (coo format)
        :param edge_type: [edge_nums]
        :return: update representations [node_nums, graph_encoder_out_dim]
        """
        if self.conv_name == 'hgt':
            return self.conv(meta_xs, node_type, edge_index, edge_type, node_position)
        elif self.conv_name == 'gcn':
            return self.conv(meta_xs, edge_index)
        elif self.conv_name == "rgcn":
            return self.conv(meta_xs, edge_index, edge_type)
        elif self.conv_name == 'gat':
            return self.conv(meta_xs, edge_index)
