import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class MultiplicativeAttention(nn.Module):
    ''' Multiplicative Attention layer '''

    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.attn_weight = nn.Linear(d_hidden, d_model)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, enc_output, ses_hidden, non_pad_mask):
        attn_vec = self.attn_weight(ses_hidden).unsqueeze(-1)

        #- Compute attention distribution and fill pad values with 0
        attn_distr = torch.bmm(enc_output, attn_vec).repeat(1, 1, enc_output.size(-1))
        attn_distr[~non_pad_mask.bool()] = -1e9
        attn_distr = self.softmax(attn_distr)
        
        return attn_distr

class DotProductAttention(nn.Module):
    ''' Vanilla Dot Product Attention layer '''

    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        assert d_model == d_hidden
    
    def forward(self, enc_output, ses_hidden, non_pad_mask):
        attn_vec = ses_hidden.unsqueeze(-1)

        #- Compute attention distribution and fill pad values with 0
        attn_distr = torch.bmm(enc_output, attn_vec).repeat(1, 1, enc_output.size(-1))
        attn_distr[~non_pad_mask] = float('-inf')
        attn_distr = self.softmax(attn_distr)

        return attn_distr
