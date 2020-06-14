import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from local import LocalAttentionEncoder
from graph import GlobalAttentionEncoder, ScaledDotProductAttention
from tasks.ubqp import reward

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def merge_muti_response(sequences):
    '''
    merge from batch * nb_slot * slot_len to batch * nb_slot * max_slot_len
    '''
    lengths = []
    for bsz_seq in sequences:
        length = [len(v) for v in bsz_seq]
        lengths.append(length)
    max_len = max([max(l) for l in lengths])
    padded_seqs = []
    for bsz_seq in sequences:
        pad_seq = []
        for v in bsz_seq:
            v = v + [0] * (max_len - len(v))
            pad_seq.append(v)
        padded_seqs.append(pad_seq)
    padded_seqs = torch.tensor(padded_seqs)
    lengths = torch.tensor(lengths)
    return padded_seqs


class Encoder_Embedding(nn.Module):
    def __init__(self,
                 input_size, hidden_size
                 ):
        super(Encoder_Embedding, self).__init__()
        self.enc = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input):
        out_put = self.enc(input)
        return out_put


class Local_Encoder_Embedding(nn.Module):
    def __init__(self,
                 input_size, const
                 ):
        super(Local_Encoder_Embedding, self).__init__()
        self.enc = nn.Conv1d(input_size, input_size * const, kernel_size=1)

    def forward(self, input):
        out_put = self.enc(input)
        return out_put


class Poniter(nn.Module):
    def __init__(self,
                 hidden_size, num_layers, dropout=0.2
                 ):
        super(Poniter, self).__init__()
        # 1119
        self.vv1 = nn.Parameter(torch.Tensor(1, 1, hidden_size))
        self.ww1 = nn.Parameter(torch.Tensor(1, hidden_size, 2 * hidden_size))
        self.vv2 = nn.Parameter(torch.Tensor(1, 1, hidden_size))
        self.ww2 = nn.Parameter(torch.Tensor(1, hidden_size, 3 * hidden_size))

    def forward(self, global_,static_hidden, dynamic_hidden):

        batch_size, hidden_size, _ = static_hidden.size()

        """1119change the simple model pointernetwork"""

        vv1 = self.vv1.expand(batch_size, -1, -1)
        ww1 = self.ww1.expand(batch_size, -1, -1)

        vv2 = self.vv2.expand(batch_size, -1, -1)
        ww2 = self.ww2.expand(batch_size, -1, -1)

        cat_1 = torch.cat((global_, dynamic_hidden), dim=1)
        attn_1 = torch.matmul(vv1, torch.tanh(torch.matmul(ww1, cat_1)))
        attns_1 = F.softmax(attn_1, dim=2)
        context = attns_1.bmm(static_hidden.permute(0, 2, 1))
        context = context.transpose(1, 2).expand_as(static_hidden)
        cat_3 = torch.cat((static_hidden, dynamic_hidden, context), dim=1)

        so = torch.matmul(vv2, torch.tanh(torch.matmul(ww2, cat_3)))
        attn = so.squeeze(1)
        return attn

class DRL4UBQP(nn.Module):
    def __init__(self, task_size, static_size, hidden_size, dynamic_size, num_layers, n_head, n_layers, k_dim,
                 v_dim, const_local, pos, update_fn=None, mask_fn=None, dropout=0.):
        super(DRL4UBQP, self).__init__()
        if dynamic_size < 1:
            raise ValueError(':param dynamic_size: must be > 0, even if the '
                             'problem has no dynamic elements')

        self.hidden_size = hidden_size
        self.update_fn = update_fn
        self.mask_fn = mask_fn
        self.g_encoder = Encoder_Embedding(static_size, hidden_size)
        self.static_encoder = Encoder_Embedding(static_size, hidden_size)
        self.dynamic_encoder = Encoder_Embedding(dynamic_size, hidden_size)
        self.global_attention = GlobalAttentionEncoder(hidden_size, n_head, n_layers, k_dim,
                                                       v_dim)

        self.pointer = Poniter(hidden_size, num_layers, dropout=0.2)
        self.pos = pos

        for p in self.parameters():
            # print(p.size())
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(
                    p)

        self.x0 = torch.zeros((1, static_size, 1), requires_grad=True, device=device)

    def forward(self, Q, Q_DIAG, Q_RC, dynamic, decoder_input=None,
                current_task=None, first_task=None, last_hh=None):
        static = torch.cat((Q_DIAG, Q_RC), dim=1)
        batch_size, input_size, sequence_size = static.size()  # b 1 m+1

        mask = torch.ones(batch_size, sequence_size).to(device)
        mask[:, 0] = 0
        tour_idx, tour_logp = [], []
        max_steps = sequence_size if self.mask_fn is None else 10000

        global_hidden = self.g_encoder(static)
        static_hidden = self.static_encoder(static)
        global_, global_mean = self.global_attention(global_hidden.permute(0, 2, 1))
        idx = 0

        x = torch.ones(batch_size, sequence_size).to(device)
        x[:, 0] = 0

        dynamic = torch.ones(x.size()).unsqueeze(1).to(device)
        dynamic[:, :,0] = 0
        for _ in range(max_steps):
            if not mask.byte().any():
                break
            x_before = x.clone()
            reward_ = reward(Q, x)
            dynamic_hidden = self.dynamic_encoder(dynamic)
            probs = self.pointer(global_,
                                          static_hidden,
                                          dynamic_hidden
                                          )
            probs = F.softmax(probs + mask.log(), dim=1)
            if self.training:
                m = torch.distributions.Categorical(probs)
                ptr = m.sample()
                while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():
                    ptr = m.sample()
                logp = m.log_prob(ptr)
            else:
                prob, ptr = torch.max(probs, 1)  # Greedy
                logp = prob.log()

            idx_xy = torch.full((ptr.data.size()), 1)
            visit_idx_xy = idx_xy.nonzero().squeeze(1).long()
            x[visit_idx_xy, ptr[visit_idx_xy]] = 0
            reward_current = reward(Q, x)
            gap = reward_current.le(reward_).float()

            if self.update_fn is not None:
                x,dynamic = self.update_fn(x, x_before, gap)
            if self.mask_fn is not None:
                mask = self.mask_fn(dynamic, x).detach()
            tour_logp.append(logp.unsqueeze(1))
            idx += 1

        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)
        tour_idx = x
        return tour_idx, tour_logp


# class Pointer()
'''
local: b x m                b x m
global : batch x m x k_dim       b x m x hidden_size
global_min : batch x embed_dim     b x m x hidden_size
static : batch x m_dim x m      b x hidden_size x m
dynamic : batch x m_dim x m      b x hidden_size x m
hidden : batch x hidden_size

hidden_size = 256
local_hidden_size = static_size x const = 4 x 40 = 160
num_layers = 1
k_dim = v_dim = 256
'''

if __name__ == '__main__':
    raise Exception('Cannot be called from main')
