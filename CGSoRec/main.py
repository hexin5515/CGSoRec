"""
Train a diffusion model for recommendation
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
parser.add_argument('-data', '--dataset', type=str, default='lastfm_unbias', help='choose the dataset')
parser.add_argument('-data_p', '--data_path', type=str, default='../datasets/lastfm_unbias_noisy/', help='load data path')
parser.add_argument('-lr', '--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
parser.add_argument('--topN', type=str, default='[5, 10, 20, 50, 100]')
parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
parser.add_argument('--cuda', type=int, default=1, help='use CUDA')
parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
parser.add_argument('--log_name', type=str, default='log', help='the log name')
parser.add_argument('--round', type=int, default=1, help='record the experiment')

# params for the model
parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
parser.add_argument('-dim', '--dims', type=str, default='[1000]', help='the dims for the DNN')
parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')

# params for diffusion
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=10, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('-noise_s', '--noise_scale', type=float, default=0.0001, help='noise scale for noise generating')
parser.add_argument('-noise_mi', '--noise_min', type=float, default=0.0005, help='noise lower bound for noise generating')
parser.add_argument('-noise_ma', '--noise_max', type=float, default=0.005, help='noise upper bound for noise generating')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')
parser.add_argument('--cond_weight', type=float, default=0, help='wieght of cond for guidence of diffusion model')
args = parser.parse_args()
print("args:", args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if args.cuda else "cpu")

print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

### DATA LOAD ###
train_path = args.data_path + 'train_list.npy'
train_tail_path = args.data_path + 'train_tail_list.npy'
valid_path = args.data_path + 'valid_list.npy'
test_path = args.data_path + 'test_list.npy'
social_path = args.data_path + 'social_list.npy'
diffusion_social_path = args.data_path + 'diffusion_social_list.npy'

train_data, train_data_v, train_data_add_v, train_data_add_h, test_y_data, n_user, n_item = data_utils.data_load_ori(train_path, test_path, social_path, train_tail_path)
train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A))
train_dataset_v = data_utils.DataDiffusion(torch.FloatTensor(train_data_v.A))
train_dataset_add_v = data_utils.DataDiffusion(torch.FloatTensor(train_data_add_v.A))
train_dataset_add_h = data_utils.DataDiffusion(torch.FloatTensor(train_data_add_h.A))
train_loader = DataLoader(train_dataset_add_v, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
test_loader = DataLoader(train_dataset_add_h, batch_size=args.batch_size, shuffle=False)

if args.tst_w_val:
    tv_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A) + torch.FloatTensor(valid_y_data.A))
    test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)

mask_tv = train_data

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
out_dims = eval(args.dims) + [n_item]  #[n_user + n_item]
in_dims = out_dims[::-1]
model = DNN(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm).to(device) # in_dims=[n_user + n_item, args.dims]

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
print("models ready.")

param_num = 0
mlp_num = sum([param.nelement() for param in model.parameters()])
diff_num = sum([param.nelement() for param in diffusion.parameters()])  # 0
param_num = mlp_num + diff_num
print("Number of all parameters:", param_num)

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

            train_len = torch.count_nonzero(batch_1, dim=1).reshape(-1, 1)
            train_len = train_len.cpu().numpy().tolist()

            prediction_main = diffusion.p_sample_train(model, batch_1, args.sampling_steps, args.sampling_noise)
            prediction_c = diffusion.p_sample_train(model, batch_2, args.sampling_steps, args.sampling_noise)

            if args.cond_weight != 0:
                for i in range(len(prediction_main)):
                    if train_len[i][0] <= 10:
                        prediction_main[i] = (1 + args.cond_weight) * prediction_c[i] - args.cond_weight * prediction_main[i]

            prediction_main[torch.FloatTensor(his_data.A) == 1] = -np.inf

            _, indices = torch.topk(prediction_main, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)

    test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)

    return test_results

best_recall, best_epoch = -100, 0
best_test_result = None
print("Start training...")
for epoch in range(1, args.epochs + 1):
    if epoch - best_epoch >= 40:
        print('-'*18)
        print('Exiting from training early')
        break

    model.train()
    start_time = time.time()

    batch_count = 0
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        batch_count += 1
        optimizer.zero_grad()
        losses = diffusion.training_losses(model, batch, args.reweight)
        loss = losses["loss"].mean()
        total_loss += loss
        loss.backward()
        optimizer.step()
    
    if epoch % 5 == 0:
        if args.tst_w_val:
            test_results = evaluate(test_twv_loader, test_y_data, mask_tv, eval(args.topN))
        else:
            test_results = evaluate(test_loader, test_y_data, mask_tv, eval(args.topN))
        evaluate_utils.print_results(None, None, test_results)

        if test_results[1][2] > best_recall: # recall@20 as selection
            best_recall, best_epoch = test_results[1][2], epoch
            best_results = test_results
            best_test_results = test_results

            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(model, '{}{}_lr{}_wd{}_bs{}_dims{}_emb{}_{}_steps{}_scale{}_min{}_max{}_sample{}_reweight{}_normal{}_{}_add_cond.pth' \
                .format(args.save_path, args.dataset, args.lr, args.weight_decay, args.batch_size, args.dims, args.emb_size, args.mean_type, \
                args.steps, args.noise_scale, args.noise_min, args.noise_max, args.sampling_steps, args.reweight, args.norm, args.log_name))
    
    print("Runing Epoch {:03d} ".format(epoch) + 'train loss {:.4f}'.format(total_loss) + " costs " + time.strftime(
                        "%H: %M: %S", time.gmtime(time.time()-start_time)))
    print('---'*18)

print('==='*18)
print("End. Best Epoch {:03d} ".format(best_epoch))
evaluate_utils.print_results(None, best_results, best_test_results)   
print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

with open(args.dataset + '_result.txt', 'a') as f:
    f.write('{}_lr{}_wd{}_bs{}_dims{}_emb{}_{}_steps{}_scale{}_min{}_max{}_sample{}_reweight{}_normal{}_{}\n'.format(args.dataset, args.lr, args.weight_decay, args.batch_size, args.dims, args.emb_size, args.mean_type, \
                args.steps, args.noise_scale, args.noise_min, args.noise_max, args.sampling_steps, args.reweight, args.norm, args.log_name))
    f.write(str(best_test_results) + '\n')





