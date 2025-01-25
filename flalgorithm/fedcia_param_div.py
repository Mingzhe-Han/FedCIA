import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.laplace import Laplace

from utils.logger import *
from utils.utils import *
from utils.metric import *

from .fedavg import Fedavg

class Fedcia(Fedavg):
    def build_model(self, config):
        model_dict = [config['model1'], config['model2'], config['model3']]
        early_stop_num = config.getint("train", "early_stop_num")
        lr = config.getfloat("train", "lr")
        agg_lr = config.getfloat("train", "agg_lr")
        weight_decay = [config.getfloat("model1", "weight_decay"), config.getfloat("model2", "weight_decay"), config.getfloat("model3", "weight_decay")]

        self.client_model = list()
        self.client_optimizer = list()
        self.agg_optimizer = list()
        self.servermap = [0] * 33 + [1] * 33 + [2] * 34
        for i in range(33):
            self.client_model.append(get_model(model_dict[0], self.dataset_list[i]))
            self.client_optimizer.append(torch.optim.Adam(params=self.client_model[i].parameters(), lr=lr))
            self.agg_optimizer.append(torch.optim.SGD(params=self.client_model[i].parameters(), lr=agg_lr, weight_decay=weight_decay[0]))

        for i in range(33):
            self.client_model.append(get_model(model_dict[1], self.dataset_list[i + 33]))
            self.client_optimizer.append(torch.optim.Adam(params=self.client_model[i + 33].parameters(), lr=lr))
            self.agg_optimizer.append(torch.optim.SGD(params=self.client_model[i + 33].parameters(), lr=agg_lr, weight_decay=weight_decay[1]))

        for i in range(34):
            self.client_model.append(get_model(model_dict[2], self.dataset_list[i + 66]))
            self.client_optimizer.append(torch.optim.Adam(params=self.client_model[i + 66].parameters(), lr=lr))
            self.agg_optimizer.append(torch.optim.SGD(params=self.client_model[i + 66].parameters(), lr=agg_lr, weight_decay=weight_decay[2]))
    
        self.server_model = [get_model(model_dict[0], self.dataset_list[0]), get_model(model_dict[1], self.dataset_list[0]), get_model(model_dict[2], self.dataset_list[0])]
        self.server_paramname = [[k for k in self.server_model[0].state_dict() if 'user' not in k], 
                        [k for k in self.server_model[1].state_dict() if 'user' not in k], 
                        [k for k in self.server_model[2].state_dict() if 'user' not in k]]
        self.early_stopper = EarlyStopper_base(num_trials=early_stop_num, save_path=self.exp_save_name, client_num=self.client_num)

    def build_global_config(self, config):
        self.epoch = config.getint("train", "epoch")
        self.update_frequence = config.getint("train", "update_frequence")

    def client_init(self):
        for i in range(self.client_num):
            private_state = {k:self.server_model[self.servermap[i]].state_dict()[k] for k in self.server_paramname[self.servermap[i]]}
            self.client_model[i].load_state_dict(private_state, strict=False)

    def client_upload(self):
        item_num = self.dataset_list[0].item_num
        self.server_filter = torch.zeros((item_num, item_num)).to(self.device)
        
        for i in range(self.client_num):
            self.client_model[i] = self.client_model[i].to(self.device)
            laplace_dist = Laplace(0, 0.001)
            noise = laplace_dist.sample((item_num, item_num)).to(self.device)
            self.server_filter += self.client_model[i].getItemSimilarity().detach() + noise
            self.client_model[i] = self.client_model[i].to('cpu')
        
        self.server_filter = self.server_filter / self.client_num

    def client_download(self):
        for i in range(self.client_num):
            self.client_model[i] = self.client_model[i].to(self.device)
            self.client_model[i].train()
            for _ in range(100):
                loss = torch.norm(self.client_model[i].getItemSimilarity() - self.server_filter, p=2)
                self.agg_optimizer[i].zero_grad()
                loss.backward()
                self.agg_optimizer[i].step()
            self.client_model[i] = self.client_model[i].to('cpu')

    def train_client_model(self):
        for i in range(self.client_num):
            self.client_model[i] = self.client_model[i].to(self.device)
            self.client_model[i].train()
            for client_epoch in range(self.update_frequence):
                for (batch_users, batch_pos, batch_neg) in self.loader_list[i]:
                    batch_users, batch_pos, batch_neg = batch_users.to(self.device), batch_pos.to(self.device), batch_neg.to(self.device)
                    loss = self.client_model[i].loss(batch_users, batch_pos, batch_neg)
                    self.client_optimizer[i].zero_grad()
                    loss.backward()
                    self.client_optimizer[i].step()
            self.client_model[i] = self.client_model[i].to('cpu')
