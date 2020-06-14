import torch
from tasks.ubqp import reward as re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def DRLBISolution(Q, tour_current):  # BI
    Q = Q.clone().cpu()
    tour_current = tour_current.clone().cpu()
    tour = torch.zeros(tour_current.size())  # b x m
    idx = 0
    for i in tour_current:  #
        current_tour = i.clone()
        improve_tour = i.clone()
        while True:
            idx_i = current_tour.eq(1).squeeze(0)
            suoyin_v = idx_i.nonzero().squeeze(1)
            W = Q[idx]
            fxj_c = 0.
            for k in suoyin_v:
                cu_x = current_tour.clone()
                cu_x[k] = 0
                c = W[:, k]
                fxj = W[k, k] + 2 * (c * cu_x).sum(0)
                if fxj <fxj_c:
                    fxj_c = fxj
                    improve_tour = cu_x.clone()
            if improve_tour.equal(current_tour):
                break
            else:
                current_tour = improve_tour.clone()
        tour[idx] = improve_tour.squeeze(0)
        idx += 1
    reward = re(Q, tour).mean().item()
    return reward
