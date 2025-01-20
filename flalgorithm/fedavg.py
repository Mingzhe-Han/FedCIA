import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.logger import *
from utils.utils import *
from utils.metric import *

from .base import Base

class Fedavg(Base):
    def build_model(self, config):
        model_dict = config['model']
        early_stop_num = config.getint("train", "early_stop_num")
        lr = config.getfloat("train", "lr")

        self.client_model = list()
        self.client_optimizer = list()
        for i in range(self.client_num):
            self.client_model.append(get_model(model_dict, self.dataset_list[i]))
            self.client_optimizer.append(torch.optim.Adam(params=self.client_model[i].parameters(), lr=lr))
        self.server_model = get_model(model_dict, self.dataset_list[0])
        self.server_paramname = [k for k in self.server_model.state_dict() if 'user' not in k]
        self.early_stopper = EarlyStopper_base(num_trials=early_stop_num, save_path=self.exp_save_name, client_num=self.client_num)

    def build_global_config(self, config):
        self.epoch = config.getint("train", "epoch")
        self.update_frequence = config.getint("train", "update_frequence")

    def client_init(self):
        for i in range(self.client_num):
            private_state = {k:self.server_model.state_dict()[k] for k in self.server_paramname}
            self.client_model[i].load_state_dict(private_state, strict=False)

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


    def client_upload(self):
        datasize_sum = 0
        aggstate = {k:0.0 for k in self.server_paramname}

        for i in range(self.client_num):
            datasize = self.dataset_list[i].trainnum
            datasize_sum += datasize
            self.client_model[i] = self.client_model[i].to(self.device)
            for k in self.server_paramname:
                aggstate[k] = aggstate[k] + self.client_model[i].state_dict()[k] * datasize
            self.client_model[i] = self.client_model[i].to("cpu")

        for k in self.server_paramname:
            aggstate[k] = aggstate[k] / datasize_sum

        self.server_model.load_state_dict(aggstate, strict=False)

    def client_download(self):
        for i in range(self.client_num):
            private_state = {k:self.server_model.state_dict()[k] for k in self.server_paramname}
            self.client_model[i].load_state_dict(private_state, strict=False)

    def model_save(self, step):
        for i in range(self.client_num): 
            torch.save(self.client_model[i].state_dict()['embedding_item.weight'], self.exp_save_name + '/clients/item_' + step + str(i) + '.pth')

    def fit(self):
        for epoch_i in range(self.epoch):
            start_time = time.time()

            # Train client model
            self.epoch_i = epoch_i
            
            # Init model
            if epoch_i == 0:
                self.client_init()
                self.logger.info(f"Finish init client model in {(time.time() - start_time):.2f}s")

            # self.model_save('1')

            self.train_client_model()
            self.logger.info(f"Finish client train in {(time.time() - start_time):.2f}s")

            # self.model_save('2')

            # Upload server from client
            self.client_upload()
            self.logger.info(f"Finish upload in {(time.time() - start_time):.2f}s")

            # Download client from server
            self.client_download()
            self.logger.info(f"Finish download in {(time.time() - start_time):.2f}s")

            # self.model_save('3')

            # Test server model and Early stop
            valid_metric = self.evaluate_valid()
            self.logger.info(f"Finish valid valid set epoch {epoch_i} in {(time.time() - start_time):.2f}s, f1: {valid_metric[2]}, mrr: {valid_metric[3]}, ndcg: {valid_metric[4]}.")

            if not self.early_stopper.is_continuable(self.client_model, valid_metric[2][0], all_accuracy=valid_metric):
                self.logger.info(f'early stop at epoch {epoch_i}')
                self.logger.info(f'validation: best f1: {self.early_stopper.best_accuracy}')
                self.client_model = self.early_stopper.model_load(self.client_model)
                break

        test_metric = self.evaluate_test()
        self.logger.info(f"f1: {test_metric[2]}, mrr: {test_metric[3]}, ndcg: {test_metric[4]}.")
        self.writer.close()