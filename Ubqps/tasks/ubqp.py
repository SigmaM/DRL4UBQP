import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_transition_time(p1, p2):
    transition_time = torch.exp(
        -0.5 * (p1 * p1 + p2 * p2))
    return transition_time


def get_reward(x, Q):
    return (x.mm(Q)).mm(x.transpose(1, 0))


class WTdataset(Dataset):
    def __init__(self, num_samples, input_size, seed=None):
        super(WTdataset, self).__init__()
        self.num_samples = num_samples

        if seed is None:
            seed = np.random.randint(11111)
        torch.manual_seed(seed)
        state = torch.zeros(num_samples, 1, input_size + 1)  # b x 1 x m+1
        Q = torch.rand(num_samples, input_size, input_size)*2-1
        # 20200608 x置0.
        x = 0.9
        Q0 = torch.rand(num_samples, input_size, input_size).ge(x).float()
        Q = Q * Q0
        del Q0
        # 20200608 x置0.

        # Q = torch.randn(num_samples, input_size, input_size)  # 20200608
        # #20200608
        # Q_abs = Q.abs()
        #
        # idx_ = 0
        # for i in Q_abs:
        #     i_max = i.max()
        #     Q[idx_] /= i_max
        #     idx_ += 1
        # del Q_abs
        # #20200608
        self.dynamic = state
        self.Q = Q

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return ([], self.dynamic[idx], self.Q[idx])

    def update_mask(self, dynamic, x):
        state = dynamic.data[:, 0]
        if state[:, 1:].eq(0).all():
            return state * 0.
        new_mask = state.ne(0)
        idx_i = 0
        for i in new_mask:
            b = i[1:].clone()
            if b.eq(0).all():
                new_mask[idx_i, 0] = 1
            idx_i += 1
        return new_mask.float()

    def update_dynamic(self, x,x_before,gap):

        x = x.cpu()
        dynamic = x.clone().cpu()
        gap = gap.cpu()
        x_before = x_before.cpu()
        idx = gap.nonzero().squeeze(0).long()
        for i in idx:
            dynamic[i, 0] = 1
            dynamic[i, 1:] = 0
            x[i] = x_before[i]
        return x.to(device),dynamic.unsqueeze(1).to(device)

def reward(Q, X):
    batch_size, seq_size = X.size()
    reward = X[:, 0].clone()
    idx = 0
    for i in range(batch_size):
        reward[idx] = get_reward(X[idx].unsqueeze(0), Q[idx])
        idx += 1
    return reward
