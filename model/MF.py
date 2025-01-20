import numpy as np
import scipy.sparse as sp
from sparsesvd import sparsesvd

import torch
from torch import nn
from .base import BasicModel

class MF(BasicModel):
    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight(config)

    def __init_weight(self, config):
        self.num_users  = self.dataset.user_num
        self.num_items  = self.dataset.item_num
        self.latent_dim = config.getint('latent_dim')
        self.weight_decay = config.getfloat('weight_decay')
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.f = nn.Sigmoid()
        self.MSEloss = nn.MSELoss()

    def loss(self, users, pos, neg):
        users_emb = self.embedding_user(users)
        pos_emb = self.embedding_item(pos)
        neg_emb = self.embedding_item(neg)
        reg_loss = (1/2) * (users_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_label = torch.ones_like(pos_scores)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_label = torch.zeros_like(neg_scores)
        loss = 1/2 * (self.MSEloss(pos_scores, pos_label) + self.MSEloss(neg_scores, neg_label))

        return loss + reg_loss * self.weight_decay

    def getUsersRating(self, users):
        users_emb = self.embedding_user.weight[users]
        items_emb = self.embedding_item.weight
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getItemSimilarity(self):
        items_emb = self.embedding_item.weight
        similarity_matrix  = torch.mm(items_emb, items_emb.t())

        return similarity_matrix