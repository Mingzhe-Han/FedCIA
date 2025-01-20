import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.logger import *
from utils.utils import *
from utils.metric import *

class Base(object):
    def __init__(self, config):
        self.build_logger(config)

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        self.logger.info(f"Use device: {self.device}")

        self.build_dataset(config)
        self.build_model(config)
        self.build_global_config(config)

    def build_logger(self, config):
        exp_name = config.get("exp", "name")
        exp_output = config.get("exp", "output")
        self.logger, self.exp_save_name = logger(exp_name, exp_output)
        self.logger.info(f"logger is saved in: {self.exp_save_name}")
        set_seeds(config.getint("exp", "random_seed"))
        configfile = open(self.exp_save_name + '/config.ini', 'w')
        config.write(configfile)
        self.writer = SummaryWriter(self.exp_save_name + '/tensorboard/')
        self.client_model_path = self.exp_save_name + '/clients/'
        os.mkdir(self.client_model_path)

    def build_dataset(self, config):
        dataset_path = config.get("dataset", "path")
        batch_size = config.getint("train", "batch_size")

        self.client_num = config.getint("dataset", "client_num")
        self.dataset_list = get_split_dataset(dataset_path, self.client_num)
        self.loader_list = list()
        for dataset in self.dataset_list:
            self.loader_list.append(DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True))

    def build_model(self, config):
        model_dict = config['model']
        early_stop_num = config.getint("train", "early_stop_num")
        lr = config.getfloat("train", "lr")

        self.client_model = list()
        self.client_optimizer = list()
        for i in range(self.client_num):
            self.client_model.append(get_model(model_dict, self.dataset_list[i]))
            self.client_optimizer.append(torch.optim.Adam(params=self.client_model[i].parameters(), lr=lr))
        self.early_stopper = EarlyStopper_base(num_trials=early_stop_num, save_path=self.exp_save_name, client_num=self.client_num)

    def build_global_config(self, config):
        self.epoch = config.getint("train", "epoch")

    def train_client_model(self):
        for i in range(self.client_num):
            self.client_model[i] = self.client_model[i].to(self.device)
            self.client_model[i].train()
            for (batch_users, batch_pos, batch_neg) in self.loader_list[i]:
                batch_users, batch_pos, batch_neg = batch_users.to(self.device), batch_pos.to(self.device), batch_neg.to(self.device)
                loss = self.client_model[i].loss(batch_users, batch_pos, batch_neg)
                self.client_optimizer[i].zero_grad()
                loss.backward()
                self.client_optimizer[i].step()
            self.client_model[i] = self.client_model[i].to('cpu')

    def evaluate_valid(self):
        predicitons = list()
        labels = list()
        for i in range(self.client_num):
            self.client_model[i].eval()
            self.client_model[i] = self.client_model[i].to(self.device)
            this_pred = self.client_model[i].getUsersRating(self.dataset_list[i].validlist)
            self.client_model[i] = self.client_model[i].to('cpu')
            trainset = torch.tensor(self.dataset_list[i].trainset.A[self.dataset_list[i].validlist] > 0)
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
            self.client_model[i].eval()
            self.client_model[i] = self.client_model[i].to(self.device)
            this_pred = self.client_model[i].getUsersRating(self.dataset_list[i].testlist)
            self.client_model[i] = self.client_model[i].to('cpu')
            trainset = torch.tensor(self.dataset_list[i].trainset.A[self.dataset_list[i].testlist] > 0)
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
        for epoch_i in range(self.epoch):
            start_time = time.time()

            # Train client model
            self.epoch_i = epoch_i
            self.train_client_model()
            self.logger.info(f"Finish client train in {(time.time() - start_time):.2f}s")

            # Test server model and Early stop
            valid_metric = self.evaluate_valid()
            self.logger.info(f"Finish valid valid set epoch {epoch_i} in {(time.time() - start_time):.2f}s, f1: {valid_metric[2]}, mrr: {valid_metric[3]}, ndcg: {valid_metric[4]}.")

            if not self.early_stopper.is_continuable(self.client_model, valid_metric[2][0]):
                self.logger.info(f'early stop at epoch {epoch_i}')
                self.logger.info(f'validation: best f1: {self.early_stopper.best_accuracy}')
                self.client_model = self.early_stopper.model_load(self.client_model)
                break

        test_metric = self.evaluate_test()
        self.logger.info(f"f1: {test_metric[2]}, mrr: {test_metric[3]}, ndcg: {test_metric[4]}.")
        self.writer.close()
