"""
Train a diffusion model for recommendation
"""

import argparse
from ast import parse
import os
import time
import numpy as np
import copy
import math

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
parser.add_argument('-data','--dataset', type=str, default='ciao_unbias', help='choose the dataset')
parser.add_argument('--data_path', type=str, default='../datasets/', help='load data path')
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--topN', type=str, default='[10,20,50,100]')
parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
parser.add_argument('--cuda', type=int, default=1, help='use CUDA')
parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
parser.add_argument('--log_name', type=str, default='log', help='the log name')

# params for diffusion
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=100, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
parser.add_argument('--noise_max', type=float, default=0.9999, help='noise upper bound for noise generating')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
parser.add_argument('-N','--N', type=int, default=10, help='steps of the forward process during inference')
parser.add_argument('-scw', '--cond_weight', type=float, default=0.5, help='wieght of cond for guidence of diffusion model')


args = parser.parse_args()

args.data_path = args.data_path + args.dataset + '/'
if args.dataset == 'dbook_unbias':
    args.steps = 5
    args.noise_scale = 1e-4
    args.noise_min = 5e-4
    args.noise_max = 5e-3
elif args.dataset == 'lastfm_unbias':
    args.steps = 10
    args.noise_scale = 1e-4
    args.noise_min = 5e-4
    args.noise_max = 5e-3
elif args.dataset == 'ciao_unbias':
    args.steps = 10
    args.noise_scale = 1e-4
    args.noise_min = 5e-4
    args.noise_max = 5e-3
else:
    raise ValueError

print("args:", args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if args.cuda else "cpu")

print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

### DATA LOAD ###
social_path = args.data_path + 'social_list.npy'
rating_path = args.data_path + 'train_list.npy'
train_tail_path = args.data_path + 'train_tail_list.npy'

social_data, social_data_add_h, n_user = data_utils.social_data_load_test(social_path, rating_path, train_tail_path)
social_dataset = data_utils.DataDiffusion(torch.FloatTensor(social_data.A))
social_dataset_add_h = data_utils.DataDiffusion(torch.FloatTensor(social_data_add_h.A))
train_loader = DataLoader(social_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
test_loader = DataLoader(social_dataset_add_h, batch_size=args.batch_size, shuffle=False)
mask_tv = social_data

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
model_path = "./saved_models/social_diffusion/"
if args.dataset == "dbook_unbias":
    model_name = "dbook_lr0.0001_mtx0_wd0.0_bs400_dims[1000]_temb10_x0_steps5_scale0.0001_min0.0005_max0.005_sample0_reweightTrue_log_epoch_100_social_diff.pth"
elif args.dataset == "lastfm_unbias":
    model_name = "lastfm_lr0.0001_mtx0_wd0.0_bs400_dims[1000]_temb10_x0_steps10_scale0.0001_min0.0005_max0.005_sample0_reweightTrue_log_epoch_100_social_diff.pth"  # The filename here contains a minor error. The actual hyperparameter 'reweight=1' is used during training.
elif args.dataset == "ciao_unbias":
    model_name = "ciao_lr0.0001_mtx0_wd0.0_bs400_dims[1000]_temb10_x0_steps10_scale0.0001_min0.0005_max0.005_sample0_reweightTrue_log_epoch_100_social_diff.pth"
model = torch.load(model_path + model_name).to(device)


print("models ready.")

def cond_fn_sdg(x, t, y):
    assert y is not None
    with torch.no_grad():
        if args.cond_weight != 0:
            target_cond_features = model(y, t)
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        x_in_features = model(x_in, t)
        
        loss_cond = F.cosine_similarity(x_in_features, target_cond_features)

        total_guidance = loss_cond * args.cond_weight

        return torch.autograd.grad(total_guidance.sum(), x_in)[0]

def evaluate(data_loader, social_data, mask_his, topN):
    model.eval()
    predict_items = []
    target_items = []
    e_idxlist = list(range(mask_his.shape[0]))
    e_N = mask_his.shape[0]

    for i in range(e_N):
        target_items.append(social_data[i, :n_user].nonzero()[1].tolist())

    for batch_idx, batch in enumerate(data_loader):
        batch_1 = batch[:, :n_user].to(device)
        batch_2 = batch[:, n_user:].to(device)

        cond_fn = cond_fn_sdg
        prediction_c = diffusion.p_sample(model, batch_1, batch_2, cond_fn=cond_fn, steps=args.sampling_steps, sampling_noise=args.sampling_noise)
        prediction_main = diffusion.p_sample(model, batch_1, batch_2, cond_fn=None, steps=args.sampling_steps, sampling_noise=args.sampling_noise)
        prediction_main = (1 - args.cond_weight) * prediction_main + args.cond_weight * prediction_c

        _, indices = torch.topk(prediction_main, topN[-1])
        indices = indices.cpu().numpy().tolist()
        predict_items.extend(indices)

    result = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)
    print(result)


test_results = evaluate(test_loader, social_data, mask_tv, eval(args.topN))







