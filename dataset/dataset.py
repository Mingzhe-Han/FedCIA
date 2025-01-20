import numpy as np
import scipy.sparse as sp

import torch
from torch.utils.data import Dataset

class BasicDataset_GSP(Dataset):
    def __init__(self, path):
        self.path = path
        self.trainset = np.load(self.path + '/train_dataset.npy')
        self.validset = np.load(self.path + '/valid_dataset.npy')
        self.testset = np.load(self.path + '/test_dataset.npy')
        self.validlist = [i for i, j in enumerate(self.validset.sum(1) > 0) if j]
        self.testlist = [i for i, j in enumerate(self.testset.sum(1) > 0) if j]
        self.getlabel()
        self.user_num = self.trainset.shape[0]
        self.item_num = self.trainset.shape[1]

    def getlabel(self):
        self.validlabel = list()
        for valid_line in self.validset[self.validlist]:
            indices = [i for i, x in enumerate(valid_line) if x > 0]
            self.validlabel.append(indices)
        self.testlabel = list()
        for test_line in self.testset[self.testlist]:
            indices = [i for i, x in enumerate(test_line) if x > 0]
            self.testlabel.append(indices)

class BasicDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.trainset = np.load(self.path + '/train_dataset.npy')
        self.validset = np.load(self.path + '/valid_dataset.npy')
        self.testset = np.load(self.path + '/test_dataset.npy')
        self.validlist = [i for i, j in enumerate(self.validset.sum(1) > 0) if j]
        self.testlist = [i for i, j in enumerate(self.testset.sum(1) > 0) if j]
        self.getlabel()
        self.user_num = self.trainset.shape[0]
        self.item_num = self.trainset.shape[1]
        self.users_D = self.trainset.sum(axis=1).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = self.trainset.sum(axis=0).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        self.trainset_matrix = self.trainset
        self.trainset = sp.csr_matrix(self.trainset)
        self.trainuser, self.trainitem = self.trainset.nonzero()
        self.trainnum = len(self.trainuser)
        self.allPos = self.getUserPosItems(list(range(self.user_num)))
        self.Graph = None
        self.S = self.UniformSample()

    def getlabel(self):
        self.validlabel = list()
        for valid_line in self.validset[self.validlist]:
            indices = [i for i, x in enumerate(valid_line) if x > 0]
            self.validlabel.append(indices)
        self.testlabel = list()
        for test_line in self.testset[self.testlist]:
            indices = [i for i, x in enumerate(test_line) if x > 0]
            self.testlabel.append(indices)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/norm_adj.npz')
                norm_adj = pre_adj_mat
            except :
                adj_mat = sp.dok_matrix((self.user_num + self.item_num, self.user_num + self.item_num), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.trainset.tolil()
                adj_mat[:self.user_num, self.user_num:] = R
                adj_mat[self.user_num:, :self.user_num] = R.T
                adj_mat = adj_mat.todok()
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                sp.save_npz(self.path + '/norm_adj.npz', norm_adj)

            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj).coalesce()
            device = 'cpu'
            if torch.cuda.is_available():
                device = 'cuda'
            self.Graph = self.Graph.to(device)
        return self.Graph

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.trainset[user].nonzero()[1])
        return posItems

    def UniformSample(self):
        user_num = self.trainnum
        users = np.random.randint(0, self.user_num, user_num)
        allPos = self.allPos
        S = []
        for i, user in enumerate(users):
            posForUser = allPos[user]
            if len(posForUser) == 0:
                continue
            posindex = np.random.randint(0, len(posForUser))
            positem = posForUser[posindex]
            while True:
                negitem = np.random.randint(0, self.item_num)
                if negitem in posForUser:
                    continue
                else:
                    break
            S.append([user, positem, negitem])
        return torch.tensor(S).long()

    def __len__(self):
        return self.S.shape[0]

    def __getitem__(self, index):
        return self.S[index][0], self.S[index][1], self.S[index][2]
