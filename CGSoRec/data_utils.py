import numpy as np
from fileinput import filename
import random
import torch
import torch.utils.data as data
import scipy.sparse as sp
import copy
import os
from torch.utils.data import Dataset
from tqdm import tqdm

def data_load_ori(train_path, test_path, social_path, train_tail_path):
    train_list = np.load(train_path, allow_pickle=True)
    train_tail_list = np.load(train_tail_path, allow_pickle=True)
    test_list = np.load(test_path, allow_pickle=True)
    social_list = np.load(social_path, allow_pickle=True)

    uid_max = 0
    iid_max = 0
    train_dict = {}
    test_dict = {}
    social_dict = {}

    for uid, iid in train_list:
        if uid not in train_dict:
            train_dict[uid] = []
        train_dict[uid].append(iid)
        if uid > uid_max:
            uid_max = uid
        if iid > iid_max:
            iid_max = iid

    for uid, iid in test_list:
        if uid not in test_dict:
            test_dict[uid] = []
        test_dict[uid].append(iid)
        if uid > uid_max:
            uid_max = uid
        if iid > iid_max:
            iid_max = iid

    for u1id, u2id in social_list:
        if u1id not in social_dict:
            social_dict[u1id] = []
        social_dict[u1id].append(u2id)
        if u1id > uid_max or u2id > uid_max:
            uid_max = max(u1id, u2id)

    
    n_user = uid_max + 1
    n_item = iid_max + 1
    print(f'user num: {n_user}')
    print(f'item num: {n_item}')

    train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]), \
        (train_list[:, 0], train_list[:, 1])), dtype='float64', \
        shape=(n_user, n_item))

    train_tail_data = sp.csr_matrix((np.ones_like(train_tail_list[:, 0]), \
        (train_tail_list[:, 0], train_tail_list[:, 1])), dtype='float64', \
        shape=(n_user, n_item)) # n_user + n_item
    

    test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
                 (test_list[:, 0], test_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # test_groundtruth
    
    social_data = sp.csr_matrix((np.ones_like(social_list[:, 0]), \
        (social_list[:, 0], social_list[:, 1])), dtype='float64', \
        shape=(n_user, n_user)) # n_user + n_item
    
    # SR_ori = social_data.dot(train_data)
    SR_ori = train_tail_data.dot(train_tail_data.T.dot(train_tail_data)) + social_data.dot(train_tail_data)
    train_data_m = train_data.toarray()
    SR = SR_ori.toarray()
    SR[train_data_m == 1] = 0
    SR[SR >= 1] = 1/SR[SR >= 1] * 0.1
    # SR[SR >= 1] /= 100.
    SR = SR + train_data_m
    SR = sp.csr_matrix(SR)

    train_data_add_v = sp.vstack([train_data, SR]).tocsr()
    train_data_add_h = sp.hstack([train_data, SR]).tocsr()
    train_data_v = sp.vstack([train_data, train_data]).tocsr()
    
    return train_data, train_data_v, train_data_add_v, train_data_add_h, test_y_data, n_user, n_item

def social_data_load(social_path):

    social_list = np.load(social_path, allow_pickle=True)

    uid_max = 0
    social_dict = {}

    for u1id, u2id in social_list:
        if u1id not in social_dict:
            social_dict[u1id] = []
        social_dict[u1id].append(u2id)
        if u1id > uid_max or u2id > uid_max:
            uid_max = max(u1id, u2id)

    n_user = uid_max + 1
    print(f'user num: {n_user}')

    social_data = sp.csr_matrix((np.ones_like(social_list[:, 0]), \
        (social_list[:, 0], social_list[:, 1])), dtype='float64', \
        shape=(n_user + n_user, n_user))

    return social_data, n_user

def social_data_load_test(social_path, rating_path, train_tail_path):

    social_list = np.load(social_path, allow_pickle=True)
    rating_list = np.load(rating_path, allow_pickle=True)
    train_tail_list = np.load(train_tail_path, allow_pickle=True)

    uid_max = 0
    iid_max = 0
    social_dict = {}
    rating_dict = {}

    for uid, iid in rating_list:
        if uid not in rating_dict:
            rating_dict[uid] = []
        rating_dict[uid].append(iid)
        if iid > iid_max:
            iid_max = iid

    n_item = iid_max + 1
    print(f'item num: {n_item}')

    for u1id, u2id in social_list:
        if u1id not in social_dict:
            social_dict[u1id] = []
        social_dict[u1id].append(u2id)
        if u1id > uid_max or u2id > uid_max:
            uid_max = max(u1id, u2id)

    n_user = uid_max + 1
    print(f'user num: {n_user}')


    social_data = sp.csr_matrix((np.ones_like(social_list[:, 0]), \
        (social_list[:, 0], social_list[:, 1])), dtype='float64', \
        shape=(n_user, n_user))
    
    train_tail_data = sp.csr_matrix((np.ones_like(train_tail_list[:, 0]), \
        (train_tail_list[:, 0], train_tail_list[:, 1])), dtype='float64', \
        shape=(n_user, n_item)) # n_user + n_item
    
    SR = train_tail_data.dot(train_tail_data.T)
    social_data_m = social_data.toarray()
    SR = SR.toarray()
    SR[social_data_m == 1] = 0
    SR[SR >= 1]/=100.
    SR = SR + social_data_m
    SR = sp.csr_matrix(SR)
    

    social_data_add_h = sp.hstack([social_data, SR]).tocsr()

    return social_data, social_data_add_h, n_user

class DataDiffusion(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        item = self.data[index]
        return item
    def __len__(self):
        return len(self.data)

