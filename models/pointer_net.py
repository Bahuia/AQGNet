import torch
import torch.nn as nn
import torch.nn.utils
from torch.nn import Parameter


class AuxiliaryPointerNet(nn.Module):

    def __init__(self, query_vec_size, src_encoding_size, attention_type='affine'):
        super(AuxiliaryPointerNet, self).__init__()

        assert attention_type in ('affine', 'dot_prod')
        if attention_type == 'affine':
            self.src_encoding_linear = nn.Linear(src_encoding_size, query_vec_size, bias=False)
            self.auxiliary_encoding_linear = nn.Linear(src_encoding_size, query_vec_size, bias=False)
        self.attention_type = attention_type

    def forward(self, src_encodings, src_context_encodings, src_token_mask, query_vec):
        """
        :param src_context_encodings: Variable(batch_size, src_sent_len, src_encoding_size)
        :param src_encodings: Variable(batch_size, src_sent_len, src_encoding_size)
        :param src_token_mask: Variable(batch_size, src_sent_len)
        :param query_vec: Variable(tgt_action_num, batch_size, query_vec_size)
        :return: Variable(tgt_action_num, batch_size, src_sent_len)
        """

        # (batch_size, 1, src_sent_len, query_vec_size)
        encodings = src_encodings.clone()
        context_encodings = src_context_encodings.clone()
        if self.attention_type == 'affine':
            encodings = self.src_encoding_linear(src_encodings)
            context_encodings = self.auxiliary_encoding_linear(src_context_encodings)
        encodings = encodings.unsqueeze(1)
        context_encodings = context_encodings.unsqueeze(1)

        # (batch_size, tgt_action_num, query_vec_size, 1)
        q = query_vec.permute(1, 0, 2).unsqueeze(3)

        # (batch_size, tgt_action_num, src_sent_len)
        weights = torch.matmul(encodings, q).squeeze(3)
        context_weights = torch.matmul(context_encodings, q).squeeze(3)

        # (tgt_action_num, batch_size, src_sent_len)
        weights = weights.permute(1, 0, 2)
        context_weights = context_weights.permute(1, 0, 2)

        if src_token_mask is not None:
            # (tgt_action_num, batch_size, src_sent_len)
            src_token_mask = src_token_mask.unsqueeze(0).expand_as(weights)
            weights.data.masked_fill_(src_token_mask.bool(), -float('inf'))
            context_weights.data.masked_fill_(src_token_mask.bool(), -float('inf'))

        sigma = 0.1
        return weights.squeeze(0) + sigma * context_weights.squeeze(0)


class PointerNet(nn.Module):
    def __init__(self, query_vec_size, src_encoding_size, attention_type='affine'):
        super(PointerNet, self).__init__()

        assert attention_type in ('affine', 'dot_prod')
        if attention_type == 'affine':
            self.src_encoding_linear = nn.Linear(src_encoding_size, query_vec_size, bias=False)

        self.attention_type = attention_type
        self.input_linear = nn.Linear(query_vec_size, query_vec_size)
        self.type_linear = nn.Linear(32, query_vec_size)
        self.V = Parameter(torch.FloatTensor(query_vec_size), requires_grad=True)
        self.tanh = nn.Tanh()
        self.context_linear = nn.Conv1d(src_encoding_size, query_vec_size, 1, 1)
        self.coverage_linear = nn.Conv1d(1, query_vec_size, 1, 1)


        nn.init.uniform_(self.V, -1, 1)

    def forward(self, src_encodings, src_token_mask, query_vec):
        """
        :param src_encodings: Variable(batch_size, src_sent_len, hidden_size * 2)
        :param src_token_mask: Variable(batch_size, src_sent_len)
        :param query_vec: Variable(tgt_action_num, batch_size, query_vec_size)
        :return: Variable(tgt_action_num, batch_size, src_sent_len)
        """

        # (batch_size, 1, src_sent_len, query_vec_size)

        if self.attention_type == 'affine':
            src_encodings = self.src_encoding_linear(src_encodings)
        src_encodings = src_encodings.unsqueeze(1)

        # (batch_size, tgt_action_num, query_vec_size, 1)
        q = query_vec.permute(1, 0, 2).unsqueeze(3)

        weights = torch.matmul(src_encodings, q).squeeze(3)

        weights = weights.permute(1, 0, 2)

        if src_token_mask is not None:
            src_token_mask = src_token_mask.unsqueeze(0).expand_as(weights)
            weights.data.masked_fill_(src_token_mask.bool(), -float('inf'))

        return weights.squeeze(0)
