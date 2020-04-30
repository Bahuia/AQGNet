import os
import sys
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

sys.path.append('..')
from utils.utils import pad
from utils.embedding import Embeddings


class LSTM(nn.Module):

    def __init__(self, d_input, d_h, n_layers=1, birnn=True, dropout=0.3):
        super(LSTM, self).__init__()

        n_dir = 2 if birnn else 1
        self.init_h = nn.Parameter(torch.Tensor(n_layers * n_dir, d_h))
        self.init_c = nn.Parameter(torch.Tensor(n_layers * n_dir, d_h))

        INI = 1e-2
        torch.nn.init.uniform_(self.init_h, -INI, INI)
        torch.nn.init.uniform_(self.init_c, -INI, INI)

        self.lstm = nn.LSTM(
            input_size=d_input,
            hidden_size=d_h,
            num_layers=n_layers,
            bidirectional=birnn,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, seqs, seq_lens=None, init_states=None):

        bs = seqs.size(0)
        bf = self.lstm.batch_first

        if not bf:
            seqs = seqs.transpose(0, 1)

        seqs = self.dropout(seqs)


        size = (self.init_h.size(0), bs, self.init_h.size(1))
        if init_states is None:
            init_states = (self.init_h.unsqueeze(1).expand(*size).contiguous(),
                           self.init_c.unsqueeze(1).expand(*size).contiguous())

        if seq_lens is not None:
            assert bs == len(seq_lens)
            sort_ind = sorted(range(len(seq_lens)), key=lambda i: seq_lens[i], reverse=True)
            seq_lens = [seq_lens[i] for i in sort_ind]
            seqs = self.reorder_sequence(seqs, sort_ind, bf)
            init_states = self.reorder_init_states(init_states, sort_ind)

            packed_seq = nn.utils.rnn.pack_padded_sequence(seqs, seq_lens)
            packed_out, final_states = self.lstm(packed_seq)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out)

            back_map = {ind: i for i, ind in enumerate(sort_ind)}
            reorder_ind = [back_map[i] for i in range(len(seq_lens))]
            lstm_out = self.reorder_sequence(lstm_out, reorder_ind, bf)
            final_states = self.reorder_init_states(final_states, reorder_ind)
        else:
            lstm_out, final_states = self.lstm(seqs)
        return lstm_out.transpose(0, 1), final_states

    def reorder_sequence(self, seqs, order, batch_first=False):
        """
        seqs: [T, B, D] if not batch_first
        order: list of sequence length
        """
        batch_dim = 0 if batch_first else 1
        assert len(order) == seqs.size()[batch_dim]
        order = torch.LongTensor(order).to(seqs.device)
        sorted_seqs = seqs.index_select(index=order, dim=batch_dim)
        return sorted_seqs

    def reorder_init_states(self, states, order):
        """
        lstm_states: (H, C) of tensor [layer, batch, hidden]
        order: list of sequence length
        """
        assert isinstance(states, tuple)
        assert len(states) == 2
        assert states[0].size() == states[1].size()
        assert len(order) == states[0].size()[1]

        order = torch.LongTensor(order).to(states[0].device)
        sorted_states = (states[0].index_select(index=order, dim=1),
                         states[1].index_select(index=order, dim=1))
        return sorted_states