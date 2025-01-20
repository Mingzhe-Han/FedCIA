import numpy as np
import scipy.sparse as sp
from sparsesvd import sparsesvd

import torch
from torch import nn
from .base import BasicModel

class NCF(BasicModel):
    def __init__(self, config, dataset):
        super().__init__()
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        self.config = config
        self.dataset = dataset
        self.__init_weight(config)

    def __init_weight(self, config):
        self.num_users  = self.dataset.user_num
        self.num_items  = self.dataset.item_num
        self.latent_dim = config.getint('latent_dim')
        self.weight_decay = config.getfloat('weight_decay')
        self.hidden_units = list(map(int, config.get('hidden_units').split(',')))
        self.embedding_user_GMF = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item_GMF = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_user_MLP = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item_MLP = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user_GMF.weight, std=0.1)
        nn.init.normal_(self.embedding_item_GMF.weight, std=0.1)
        nn.init.normal_(self.embedding_user_MLP.weight, std=0.1)
        nn.init.normal_(self.embedding_item_MLP.weight, std=0.1)
        self.dnn_layer = self.build_dnn(2 * self.latent_dim, self.hidden_units)
        self.fc_predict = torch.nn.Linear(self.latent_dim + self.hidden_units[-1], 1)
        self.f = nn.Sigmoid()
        self.MSEloss = nn.BCELoss()

    def build_dnn(self, input_dim, hidden_units):
        dnn_layers = list()
        hidden_units = [input_dim] + hidden_units
        for i in range(len(hidden_units) - 1):
            dnn_layers.append(torch.nn.Linear(hidden_units[i], hidden_units[i + 1]))
            dnn_layers.append(torch.nn.ReLU())
        return torch.nn.Sequential(*dnn_layers)

    def forward(self, user, item):
        users_emb_g = self.embedding_user_GMF.weight[user]
        item_emb_g = self.embedding_item_GMF.weight[item]
        out_GMF = users_emb_g * item_emb_g

        users_emb_m = self.embedding_user_MLP.weight[user]
        item_emb_m = self.embedding_item_MLP.weight[item]
        out_MLP = self.dnn_layer(torch.cat((users_emb_m, item_emb_m), -1))

        return self.f(self.fc_predict(torch.cat((out_GMF, out_MLP), -1)))

    def loss(self, users, pos, neg):
        pos_predict = self.forward(users, pos)
        pos_label = torch.ones_like(pos_predict)
        neg_predict = self.forward(users, neg)
        neg_label = torch.zeros_like(neg_predict)
        loss = 1/2 * (self.MSEloss(pos_predict, pos_label) + self.MSEloss(neg_predict, neg_label))

        return loss

    def getUsersRating(self, users):
        self.eval()
        item_all = torch.tensor(range(self.num_items))
        user_scores = list()

        # eval_batch = 1000        
        # for u in users:
        #     scores = list()
        #     for _ in range(self.num_items // eval_batch):
        #         scores.append(self.forward(torch.tensor(u).repeat(eval_batch), item_all[_ * eval_batch: (_ + 1) * eval_batch]).reshape(-1).detach().to("cpu"))
        #     scores.append(self.forward(torch.tensor(u).repeat(self.num_items % eval_batch), item_all[(_ + 1) * eval_batch:]).reshape(-1).detach().to("cpu"))
        #     user_scores.append(torch.cat(scores, dim=-1).reshape(1, -1))

        if users == []:
            scores = self.forward(torch.tensor(0).repeat(self.num_items), item_all).reshape(1, -1).detach().to("cpu")
            return scores[users]
        for u in users:
            user_scores.append(self.forward(torch.tensor(u).repeat(self.num_items), item_all).reshape(1, -1).detach().to("cpu"))

        return torch.cat(user_scores, dim=0)