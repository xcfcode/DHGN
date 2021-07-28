""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch

from onmt.utils.misc import sequence_mask


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, lengths, recover, node_nums, max_node_num, max_node_len,
                batch_edge_index, batch_edge_type, batch_node_type, batch_node_position, tgt,
                bptt=False, with_align=False):
        dec_in = tgt[:-1]  # exclude last target from inputs

        """hierarchical encoder：node encoder + graph encoder"""
        _, memory_bank, lengths = self.encoder(src, lengths, recover, node_nums, max_node_num,
                                               max_node_len, batch_edge_index, batch_edge_type,
                                               batch_node_type,
                                               batch_node_position)  # memory_bank  [seq_len, node_batch, rnn_size]

        node_lengths = lengths
        """reconstruct src (node batch to original batch) """
        # first recover src order
        src = src.transpose(0, 1).contiguous()[recover].squeeze(-1)  # [node_batch, seq_len]
        # src = src.squeeze(-1).t().contiguous()
        src_items = torch.split(src, max_node_num, dim=0)
        length_items = torch.split(lengths, max_node_num, dim=0)
        orig_lengths = list(map(torch.sum, length_items))
        max_len = max(orig_lengths)
        src_res = torch.ones(len(src_items), max_len, dtype=torch.float).to(
            src.device)  # [true batch, max_orig_seq_len]
        for index, (src_item, length_item, node_num) in enumerate(zip(src_items, length_items, node_nums)):
            temp_res = []
            for one, length in zip(src_item, length_item):
                if length != 0:
                    one = one[:length]
                    temp_res.append(one)
            item = torch.cat(temp_res, dim=0)
            src_res[index, :item.size(0)] = item
        src = src_res.transpose(0, 1).contiguous().unsqueeze(-1).to(src.device)  # [max_orig_seq_len,true_batch,1]

        """reconstruct length (node batch to original batch) """
        lengths = torch.tensor(orig_lengths, dtype=torch.float).to(src.device)

        """reconstruct memory_bank (node batch to original batch) """
        memory_bank = memory_bank.transpose(0, 1).contiguous()  # [node_batch, max_node_len , rnn_size]
        memory_bank_items = torch.split(memory_bank, max_node_num, dim=0)
        memory_bank_res = torch.zeros(len(memory_bank_items), max_len, memory_bank.size(-1)).to(
            src.device)  # [true_batch, max_orig_seq_len rnn_size ]
        for index, (memory_bank_item, length_item) in enumerate(zip(memory_bank_items, length_items)):
            # memory_bank_item[max_node_num,max_node_len,dim]
            temp_res = []
            for one, length in zip(memory_bank_item, length_item):
                if length != 0:
                    one = one[:length]
                    temp_res.append(one)
            item = torch.cat(temp_res, dim=0)
            memory_bank_res[index, :item.size(0)] = item
        memory_bank = memory_bank_res.transpose(0, 1).contiguous().to(src.device)  # [max_orig_len, true_batch, dim]

        """reconstruct encoder state, used to init decoder state"""
        mask = sequence_mask(lengths).float()
        mask = mask / lengths.unsqueeze(1).float()
        encoder_hidden = torch.bmm(mask.unsqueeze(1), memory_bank.transpose(0, 1)).transpose(0,
                                                                                             1)  # [1, true_batch, rnn_size]

        enc_state = (encoder_hidden, encoder_hidden)

        """decoder"""
        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(dec_in, memory_bank,
                                      memory_lengths=lengths,
                                      with_align=with_align)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
