"""
Train a diffusion model for recommendation
"""

import argparse
from ast import parse
import os
import time
import numpy as np
import copy
from sklearn.metrics import hamming_loss

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import models.gaussian_diffusion as gd
from models.DNN import DNN
import evaluate_utils
import data_utils
from copy import deepcopy
import scipy.sparse as sp
import csv

import random
random_seed = 1
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) # gpu
np.random.seed(random_seed) # numpy
random.seed(random_seed) # random and transforms
torch.backends.cudnn.deterministic=True # cudnn
def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

parser = argparse.ArgumentParser()
parser.add_argument('-data','--dataset', type=str, default='lastfm_unbias', help='choose the dataset')
parser.add_argument('--data_path', type=str, default='../datasets/', help='load data path')
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--topN', type=str, default='[5, 10, 20, 50, 100]')
parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
parser.add_argument('--cuda', type=int, default=1, help='use CUDA')
parser.add_argument('--gpu', type=str, default='1', help='gpu card ID')
parser.add_argument('--log_name', type=str, default='log', help='the log name')

# params for diffusion
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=5, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for noise generating')
parser.add_argument('--sampling_noise', type=bool, default=True, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
parser.add_argument('-cw', '--cond_weight', type=float, default=0.5, help='wieght of cond for guidence of diffusion model')
parser.add_argument('-scw', '--s_cond_weight', type=float, default=0, help='wieght of cond for guidence of diffusion model')
parser.add_argument('-N', '--N', type=int, default=0, help='steps of the forward process during inference')

args = parser.parse_args()

args.data_path = args.data_path + args.dataset + '/'

if args.dataset == 'dbook_unbias':
    args.steps = 10
    args.noise_scale = 0.0001
    args.noise_min = 0.0005
    args.noise_max = 0.005
elif args.dataset == 'lastfm_unbias':
    args.steps = 10
    args.noise_scale = 0.0001
    args.noise_min = 0.0005
    args.noise_max = 0.005
elif args.dataset == 'ciao_unbias':
    args.steps = 10
    args.noise_scale = 0.0001
    args.noise_min = 0.0005
    args.noise_max = 0.005
else:
    raise ValueError

print("args:", args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if args.cuda else "cpu")

print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

### DATA LOAD ###
train_path = args.data_path + 'train_list.npy'
train_tail_path = args.data_path + 'train_tail_list.npy'
valid_path = args.data_path + 'valid_list.npy'
test_path = args.data_path + 'test_list.npy'
test_hot_path = args.data_path + 'test_hot_list.npy'
test_tail_path = args.data_path + 'test_tail_list.npy'
social_path = args.data_path + 'social_list.npy' #denoise_90_social_list
diffusion_social_path = args.data_path + 'diffusion_social_list.npy'

train_data, train_data_v, train_data_add_v, train_data_add_h, test_y_data, n_user, n_item = data_utils.data_load_ori(train_path, test_path, social_path, train_tail_path)
train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A))
train_dataset_add_v = data_utils.DataDiffusion(torch.FloatTensor(train_data_add_v.A))
train_dataset_add_h = data_utils.DataDiffusion(torch.FloatTensor(train_data_add_h.A))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
test_loader = DataLoader(train_dataset_add_h, batch_size=args.batch_size, shuffle=False)


if args.tst_w_val:
    tv_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data_add_h.A) + torch.FloatTensor(valid_y_data_h.A))
    test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)


mask_tv = train_data

print('data ready.')


### CREATE DIFFUISON ###
if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError("Unimplemented mean type %s" % args.mean_type)

diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, \
        args.noise_scale, args.noise_min, args.noise_max, args.steps, device)
diffusion.to(device)

### CREATE DNN ###
model_path = "./saved_models/"
if args.dataset == "dbook_unbias":
    model_name = "dbook_unbias_lr0.0001_wd0_bs400_dims[1000]_emb10_x0_steps10_scale0.0001_min0.0005_max0.005_sample0_reweightTrue_normalFalse_log_add_cond_tail.pth"
elif args.dataset == "lastfm_unbias":
    model_name = "lastfm_unbias_lr0.0001_wd0_bs400_dims[1000]_emb10_x0_steps10_scale0.0001_min0.0005_max0.005_sample0_reweightTrue_normalFalse_log_add_cond_tail.pth"
elif args.dataset == 'ciao_unbias':
    model_name = "ciao_unbias_lr0.0001_wd0_bs400_dims[1000]_emb10_x0_steps10_scale0.0001_min0.0005_max0.005_sample0_reweightTrue_normalFalse_log.pth"
model = torch.load(model_path + model_name).to(device)

print("models ready.")

pdist = nn.PairwiseDistance(p=2)
loss_fun = nn.L1Loss()

def cond_fn_sdg(x, t, y):
    pass
def evaluate(data_loader, data_te, mask_his, topN):
    model.eval()
    e_idxlist = list(range(mask_his.shape[0]))
    e_N = mask_his.shape[0]
    predict_items = []
    target_items = []
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            his_data = mask_his[e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]]
            
            batch_1 = batch[:, :n_item].to(device)
            batch_2 = batch[:, n_item:].to(device)

            cond_fn = cond_fn_sdg
            prediction_c = diffusion.p_sample(model, batch_1, batch_2, cond_fn=cond_fn, steps=args.sampling_steps, sampling_noise=args.sampling_noise)
            prediction_main = diffusion.p_sample(model, batch_1, batch_2, cond_fn=None, steps=args.sampling_steps, sampling_noise=args.sampling_noise)
            prediction_main = (1 - args.cond_weight) * prediction_main + args.cond_weight * prediction_c

            prediction_main[his_data.nonzero()] = -np.inf

            _, indices = torch.topk(prediction_main, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)

    test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)
    evaluate_utils.print_results(None, None, test_results)
    


test_results = evaluate(test_loader, test_y_data, mask_tv, eval(args.topN))






