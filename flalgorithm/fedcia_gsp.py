import time
import random
import scipy.sparse as sp
import multiprocessing

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.laplace import Laplace

from utils.logger import *
from utils.utils import *
from utils.metric import *

from .base_gsp import Base_gsp

class Fedcia(Base_gsp):
    def build_global_config(self, config):
        self.agg_p = config.getfloat("train", "agg_p")

    def train_client_model(self):
        item_num = self.dataset_list[0].item_num
        all_server_filter = torch.zeros((item_num, item_num))
        for i in range(self.client_num):
            print(i + 1)
            user = torch.tensor(self.dataset_list[i].trainset).float()
            self.client_model[i].computer(user)
            laplace_dist = Laplace(0, 0.001)
            noise = laplace_dist.sample((item_num, item_num))
            all_server_filter += self.client_model[i].filter + noise

        all_server_filter = all_server_filter / self.client_num

        self.client_prediction = list()
        for i in range(self.client_num):
            user = torch.tensor(self.dataset_list[i].trainset).float()
            local_result = self.client_model[i].getUsersRating(user)
            self.client_model[i].filter = all_server_filter
            agg_result = self.client_model[i].getUsersRating(user)
            self.client_prediction.append(self.agg_p * agg_result + local_result)
