import numpy as np
import scipy.sparse as sp
from sparsesvd import sparsesvd

import torch
from torch import nn
from .base import BasicModel

class MLP(BasicModel):
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
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.dnn_layer = self.build_dnn(2 * self.latent_dim, self.hidden_units)
        self.f = nn.Sigmoid()
        self.MSEloss = nn.BCELoss()

    def build_dnn(self, input_dim, hidden_units):
        dnn_layers = list()
        hidden_units = [input_dim] + hidden_units
        for i in range(len(hidden_units) - 1):
            dnn_layers.append(torch.nn.Linear(hidden_units[i], hidden_units[i + 1]))
            dnn_layers.append(torch.nn.ReLU())
        dnn_layers.append(torch.nn.Linear(self.hidden_units[-1], 1))
        return torch.nn.Sequential(*dnn_layers)

    def forward(self, user, item):
        users_emb_m = self.embedding_user.weight[user]
        item_emb_m = self.embedding_item.weight[item]
        out = self.dnn_layer(torch.cat((users_emb_m, item_emb_m), -1))

        return self.f(out)

    def loss(self, users, pos, neg):
        users_emb = self.embedding_user.weight[users]
        pos_emb = self.embedding_item.weight[pos]
        neg_emb = self.embedding_item.weight[neg]
        pos_scores = self.forward(users, pos)
        pos_label = torch.ones_like(pos_scores)
        neg_scores = self.forward(users, neg)
        neg_label = torch.zeros_like(neg_scores)
        
        loss = 1/2 * (self.MSEloss(pos_scores, pos_label) + self.MSEloss(neg_scores, neg_label))
        reg_loss = (1/2) * (users_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / float(len(users))
        for param in self.dnn_layer.parameters():
            reg_loss += torch.sum(param ** 2)

        return loss + reg_loss * self.weight_decay

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

    def getItemSimilarity(self):
        items_emb = self.embedding_item.weight
        similarity_matrix = torch.mm(items_emb, items_emb.t())

        return similarity_matrix