import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.logger import *
from utils.utils import *
from utils.metric import *

from .base import Base

class Base_gsp(Base):
    def build_dataset(self, config):
        dataset_path = config.get("dataset", "path")
        batch_size = config.getint("train", "batch_size")
        self.client_num = config.getint("dataset", "client_num")
        self.dataset_list = get_split_dataset_GSP(dataset_path, self.client_num)

    def build_model(self, config):
        model_dict = config['model']
        self.client_model = list()
        for i in range(self.client_num):
            self.client_model.append(get_model(model_dict, None))

    def build_global_config(self, config):
        pass

    def train_client_model(self):
        for i in range(self.client_num):
            print(i + 1)
            user = torch.tensor(self.dataset_list[i].trainset).float()
            self.client_model[i].computer(user)

        self.client_prediction = list()
        for i in range(self.client_num):
            user = torch.tensor(self.dataset_list[i].trainset).float()
            self.client_prediction.append(self.client_model[i].getUsersRating(user))

    def evaluate_valid(self):
        predicitons = list()
        labels = list()
        for i in range(self.client_num):
            this_pred = self.client_prediction[i][self.dataset_list[i].validlist]
            trainset = torch.tensor(self.dataset_list[i].trainset[self.dataset_list[i].validlist] > 0)
            this_pred[trainset] = - 999999999.0
            prediction = torch.topk(this_pred, k = 10)[1].cpu().numpy()
            label = self.dataset_list[i].validlabel
            predicitons.extend(prediction)
            labels.extend(self.dataset_list[i].validlabel)
        res = calculate_all(labels, predicitons, [10])
        return res

    def evaluate_test(self):
        predicitons = list()
        labels = list()
        for i in range(self.client_num):
            this_pred = self.client_prediction[i][self.dataset_list[i].testlist]
            trainset = torch.tensor(self.dataset_list[i].trainset[self.dataset_list[i].testlist] > 0)
            validset = torch.tensor(self.dataset_list[i].validset[self.dataset_list[i].testlist] > 0)
            this_pred[trainset] = - 999999999.0
            this_pred[validset] = - 999999999.0
            prediction = torch.topk(this_pred, k = 10)[1].cpu().numpy()
            label = self.dataset_list[i].testlabel
            predicitons.extend(prediction)
            labels.extend(self.dataset_list[i].testlabel)
        res = calculate_all(labels, predicitons, [10])
        return res

    def fit(self):
        start_time = time.time()

        self.train_client_model()

        valid_metric = self.evaluate_valid()
        self.logger.info(f"f1: {valid_metric[2]}, mrr: {valid_metric[3]}, ndcg: {valid_metric[4]}.")
    
        test_metric = self.evaluate_test()
        self.logger.info(f"f1: {test_metric[2]}, mrr: {test_metric[3]}, ndcg: {test_metric[4]}.")
        self.writer.close()
