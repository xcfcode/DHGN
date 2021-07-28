# -*- coding: utf-8 -*-
# @Time    : 2020/3/17 8:33 下午
# @Author  : Xiachong Feng
# @File    : hier_encoder.py
# @Software: PyCharm
import torch
from torch import nn
import numpy as np

from onmt.encoders.graph_encoder import GraphEncoder
from onmt.encoders.transformer import TransformerEncoder
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.cnn_encoder import CNNEncoder
from onmt.encoders.mean_encoder import MeanEncoder

from onmt.utils.misc import sequence_mask

str2node_enc = {"rnn": RNNEncoder, "brnn": RNNEncoder, "cnn": CNNEncoder,
                "transformer": TransformerEncoder, "mean": MeanEncoder}


class HierEncoder(nn.Module):
    """
    Hierarchical Encoder, including node encoder and graph encoder
    node encoder encodes a sequence of words
    graph encoder updates node representations
    """

    def __init__(self, opt, embeddings=None):
        super(HierEncoder, self).__init__()
        self.embeddings = embeddings
        self.position_emb = UtterancePositionEmbedding()
        self.node_encoder = str2node_enc[opt.node_encoder].from_opt(opt, embeddings)
        self.graph_encoder = GraphEncoder.from_opt(opt)
        self.bridge = nn.Linear(opt.rnn_size + opt.graph_in_dim + 10, opt.rnn_size)

    @classmethod
    def from_opt(cls, opt, embeddings=None):
        return cls(opt, embeddings)

    def forward(self, src, lengths, recover, node_nums, max_node_num, max_node_len,
                batch_edge_index, batch_edge_type, batch_node_type, batch_node_position):
        node_batch = src.size(1)

        """node encoder"""
        encoder_final, memory_bank, lengths = self.node_encoder(src,
                                                                lengths)  # memory bank [seq_len, node_batch, rnn_size]
        encoder_hidden = encoder_final[0]
        encoder_cell = encoder_final[1]

        """recover"""
        memory_bank = memory_bank.transpose(0, 1).contiguous()[recover].transpose(0,
                                                                                  1)  # [seq_len, node_batch, rnn_size]
        lengths = lengths[recover]  # node_batch
        encoder_hidden = encoder_hidden.transpose(0, 1).contiguous()[recover].transpose(0,
                                                                                        1)  # [1, node_batch, rnn_size]
        encoder_cell = encoder_cell.transpose(0, 1).contiguous()[recover].transpose(0,
                                                                                    1)  # [1, node_batch, rnn_size]

        """get node feature"""
        node_feature = encoder_hidden.squeeze(0)  # [node_batch,rnn_size]

        """graph encoder"""
        update_node_features = self.graph_encoder(node_feature, batch_node_type, batch_edge_index,
                                                  batch_edge_type,
                                                  batch_node_position)  # [node_batch, graph_encoder_out_dim]

        """add position feature"""
        update_node_features = self.position_emb(update_node_features,
                                                 batch_node_position)  # [node_batch, graph_encoder_out_dim+10]

        """expand node representation to every word"""
        update_node_features = update_node_features.repeat(1, max_node_len) \
            .view(node_batch, max_node_len, -1).transpose(0,
                                                          1).contiguous()  # [seq_len, node_batch, graph_encoder_out_dim]

        """concat word and node representation"""
        update_memory_bank = torch.cat((memory_bank, update_node_features),
                                       dim=2)  # [seq_len, node_batch, graph_encoder_out_dim + rnn_size + 10]

        """transform to the same dim with decoder"""
        update_memory_bank = self.bridge(update_memory_bank)  # [seq_len, node_batch, rnn_size]

        return (encoder_hidden, encoder_cell), update_memory_bank, lengths


class UtterancePositionEmbedding(nn.Module):
    def __init__(self, max_len=36):
        super().__init__()
        self.emb = nn.Embedding(max_len, 10)

    def forward(self, x, position):
        return torch.cat((x, self.emb(position)), dim=1)
