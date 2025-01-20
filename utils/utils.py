import math
import os
import numpy as np

from dataset.dataset import *
from model import *

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def get_split_dataset(dataset_path, split_number):
    if split_number == 1:
        return [BasicDataset(dataset_path)]

    dataset_list = list()
    for i in range(split_number):
        path = dataset_path + f'_{split_number}/{i}'
        dataset_list.append(BasicDataset(path))
    return dataset_list

def get_split_dataset_GSP(dataset_path, split_number):
    if split_number == 1:
        return [BasicDataset_GSP(dataset_path)]

    dataset_list = list()
    for i in range(split_number):
        path = dataset_path + f'_{split_number}/{i}'
        dataset_list.append(BasicDataset_GSP(path))
    return dataset_list

def get_model(model_dict, dataset):
    name = model_dict['name']
    if name == 'GFCF':
        return GFCF(model_dict)
    elif name == 'MF':
        return MF(model_dict, dataset)
    elif name == 'NCF':
        return NCF(model_dict, dataset)
    elif name == 'MLP':
        return MLP(model_dict, dataset)
    elif name == 'LightGCN':
        return LightGCN(model_dict, dataset)
    else:
        raise ValueError(("No model " + model_name + " is found."))

def dataset_split(trainset, testset, client_num):
    batch_num = math.floor(trainset.shape[0] / client_num)
    trainset_list = []
    testset_list = []
    test_user_list = []
    for i in range(client_num):
        if i == client_num - 1:
            this_trainset = trainset[i * batch_num:]
            this_testset = testset[i * batch_num:]
        else:
            this_trainset =  trainset[i * batch_num: (i + 1) * batch_num]
            this_testset = testset[i * batch_num: (i + 1) * batch_num]
        this_test_user_list = [i for i, j in enumerate(this_testset.sum(1) > 0) if j]
        trainset_list.append(this_trainset)
        testset_list.append(this_testset)
        test_user_list.append(this_test_user_list)
    return trainset_list, testset_list, test_user_list

class EarlyStopper_base(object):
    def __init__(self, num_trials, save_path, client_num):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path
        self.accuracy_list = []
        self.client_num = client_num
        self.save_path = save_path + '/best_model/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def is_continuable(self, model, accuracy, all_accuracy = None):
        if all_accuracy is not None:
            self.accuracy_list.append(all_accuracy)
        else:
            self.accuracy_list.append(accuracy)
        import pickle
        with open(self.save_path + '/acc_list', 'wb') as f:
            pickle.dump(self.accuracy_list, f)
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            self.model_save(model)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

    def model_save(self, model_list):
        for i in range(self.client_num):
            torch.save(model_list[i].state_dict(), self.save_path + '/best_client_model_' + str(i) + '.pth')

    def model_load(self, model_list):
        for i in range(self.client_num):
            model_list[i].load_state_dict(torch.load(self.save_path + '/best_client_model_' + str(i) + '.pth'))
        return model_list
