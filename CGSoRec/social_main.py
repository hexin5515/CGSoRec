"""
Train a social diffusion model for ...
"""

import argparse
from ast import parse
import os
import time
import numpy as np
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import models.gaussian_diffusion as gd
from models.DNN import DNN
from models.DNN_Condition import DNN_Cond
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
parser.add_argument('--dataset', type=str, default='lastfm_unbias', help='choose the dataset')
parser.add_argument('--data_path', type=str, default='../datasets/lastfm_unbias_noisy/', help='load data path')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]')
parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
parser.add_argument('--cuda', type=int, default=1, help='use CUDA')
parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
parser.add_argument('--save_path', type=str, default='./saved_models/social_diffusion/', help='save model path')
parser.add_argument('--log_name', type=str, default='log', help='the log name')
parser.add_argument('--round', type=int, default=1, help='record the experiment')

# params for the model
parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
parser.add_argument('--dims', type=str, default='[1000]', help='the dims for the DNN')
parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')

# params for diffusion
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=10, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=1e-4, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=5e-4, help='noise lower bound for noise generating')
parser.add_argument('--noise_max', type=float, default=5e-3, help='noise upper bound for noise generating')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')

args = parser.parse_args()
print("args:", args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if args.cuda else "cpu")

print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

### DATA LOAD ###
social_path = args.data_path + 'social_list.npy'

social_data, n_user = data_utils.social_data_load(social_path)
social_dataset = data_utils.DataDiffusion(torch.FloatTensor(social_data.A))
train_loader = DataLoader(social_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
test_loader = DataLoader(social_dataset, batch_size=args.batch_size, shuffle=False)


print('data ready.')


### Build Gaussian Diffusion ###
if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError("Unimplemented mean type %s" % args.mean_type)

diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, \
        args.noise_scale, args.noise_min, args.noise_max, args.steps, device).to(device)

### Build MLP ###
out_dims = eval(args.dims) + [n_user]  #[n_user + n_item]
in_dims = out_dims[::-1]
model = DNN(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm).to(device) # in_dims=[n_user + n_item, args.dims]

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
print("models ready.")

param_num = 0
mlp_num = sum([param.nelement() for param in model.parameters()])
diff_num = sum([param.nelement() for param in diffusion.parameters()])  # 0
param_num = mlp_num + diff_num
print("Number of all parameters:", param_num)

best_recall, best_epoch = -100, 0
best_test_result = None
print("Start training...")
for epoch in range(1, args.epochs + 1):

    model.train()
    start_time = time.time()

    batch_count = 0
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)

        batch_count += 1
        optimizer.zero_grad()
        losses = diffusion.social_training_losses(model, batch)
        loss = losses["loss"].mean()
        total_loss += loss
        loss.backward()
        optimizer.step()
    
    if epoch == 100:
        torch.save(model, '{}{}_lr{}_mt{}_wd{}_bs{}_dims{}_temb{}_{}_steps{}_scale{}_min{}_max{}_sample{}_reweight{}_{}_epoch_{}_social_diff.pth' \
            .format(args.save_path, args.dataset, args.lr, args.mean_type, args.weight_decay, args.batch_size, args.dims, args.emb_size, args.mean_type, \
            args.steps, args.noise_scale, args.noise_min, args.noise_max, args.sampling_steps, args.reweight, args.log_name, epoch))
    
    print("Runing Epoch {:03d} ".format(epoch) + 'train loss {:.4f}'.format(total_loss) + " costs " + time.strftime(
                        "%H: %M: %S", time.gmtime(time.time()-start_time)))
    print('---'*18)

print('==='*18)
print("End. Best Epoch {:03d} ".format(best_epoch))   
print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))





