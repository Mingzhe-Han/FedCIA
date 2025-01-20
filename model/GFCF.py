import numpy as np
import scipy.sparse as sp

import torch
from torch import nn
from .base import BasicModel

class GFCF(BasicModel):
    def __init__(self, config):
        self.svd_number = config.getint('svd_number')
        self.wp = config.getfloat('wp')

    def computer(self, data):
        with torch.no_grad():
            rowsum = data.sum(dim=1)
            d_inv = torch.pow(rowsum, -0.5)
            d_inv[torch.isinf(d_inv)] = 0.0
            d_mat = torch.diag(d_inv)
            norm_adj = torch.mm(d_mat, data)

            colsum = data.sum(dim=0)
            d_inv = torch.pow(colsum, -0.5)
            d_inv[torch.isinf(d_inv)] = 0.0
            d_mat_i = torch.diag(d_inv)
            d_inv[d_inv == 0] = float('inf')
            d_mat_i_inv = torch.diag(1 / d_inv)
            norm_adj = torch.mm(norm_adj, d_mat_i)

            ut, s, vt = torch.linalg.svd(norm_adj)
            
            global_filter = torch.mm(norm_adj.T, norm_adj)
            idealfilter = torch.mm(torch.mm(d_mat_i, vt[:self.svd_number].T), torch.mm(vt[:self.svd_number], d_mat_i_inv))
            self.filter = global_filter + self.wp * idealfilter

    def getUsersRating(self, data):
        self.P = torch.mm(data, self.filter)
        return self.P
