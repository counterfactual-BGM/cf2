import os

import numpy as np
import argparse
from pyro.nn.dense_nn import DenseNN
from torch import nn
import pyro.distributions.transforms as T
import pyro.distributions as dist
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
from torch.distributions.transformed_distribution import TransformedDistribution


parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="root directory")
parser.add_argument("--seed", type=int, help="random seed")
parser.add_argument("--n_bins", type=int, help="root directory", default=16)
parser.add_argument("--hw", type=int, help="root directory", default=64)
parser.add_argument("--batch_size", type=int, help="log batch size", default=20)
parser.add_argument("--device", type=str, help="Compute device", default='cuda:0')
args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
DEVICE = torch.device(args.device)
NUM_EPOCHS = 1000
BATCH_SIZE = 2 ** args.batch_size

training_data = np.load(f'{args.dir}markovian/training_data.npy')
rnd_locs = np.load(f'{args.dir}markovian/locs.npy')
rnd_scales = np.load(f'{args.dir}markovian/scales.npy')
rnd_cf_data = np.load(f'{args.dir}markovian/cf_data.npy')
base_mse = np.load(f'{args.dir}markovian/cf_base_mse.npy')
rnd_cf_data[:, 0:6] = (rnd_cf_data[:, 0:6] - rnd_locs[0]) / rnd_scales[0]

data_size = training_data.shape[0]
start_indices = np.arange(0, data_size, BATCH_SIZE)
end_indices = np.append(start_indices[1:], [data_size])
param_dims = [args.n_bins, args.n_bins, args.n_bins - 1, args.n_bins]
hypernet_t = DenseNN(input_dim=2, hidden_dims=[args.hw, args.hw], param_dims=param_dims, nonlinearity=nn.ReLU()).to(device=DEVICE)
hypernet_u = DenseNN(input_dim=1, hidden_dims=[args.hw, args.hw], param_dims=param_dims, nonlinearity=nn.ReLU()).to(device=DEVICE)
modules = torch.nn.ModuleList([hypernet_u, hypernet_t])
optimizer = torch.optim.Adam(modules.parameters())
model_spec = f'bins{args.n_bins}_hw{args.hw}_seed{args.seed}'
writer = SummaryWriter(log_dir=f"{args.dir}markovian/logs/{model_spec}")
dist_base = dist.Normal(torch.zeros(1, dtype=torch.float32, device=DEVICE),
                        torch.ones(1, dtype=torch.float32, device=DEVICE))
cs_transform_t = T.ConditionalSpline(nn=hypernet_t, input_dim=1, count_bins=args.n_bins, bound=3.5, order='linear')
cs_transform_u = T.ConditionalSpline(nn=hypernet_u, input_dim=1, count_bins=args.n_bins, bound=3.5, order='linear')
affine_transform = T.AffineTransform(loc=rnd_locs[1], scale=rnd_scales[1])
u_given_rtt_dist = dist.ConditionalTransformedDistribution(dist_base, [cs_transform_u])
actions = torch.as_tensor((training_data[:, 0:1] - rnd_locs[0]) / rnd_scales[0], dtype=torch.float32, device=DEVICE)
rtts = torch.as_tensor((training_data[:, 2:3] - rnd_locs[2]) / rnd_scales[2], dtype=torch.float32, device=DEVICE)

obs_tensors = torch.as_tensor(training_data[:, 1:2], dtype=torch.float32, device=DEVICE)
context_tensors = torch.cat([actions, rtts], dim=1)

cf_actions = torch.as_tensor(rnd_cf_data[:, 0:6], dtype=torch.float32, device=DEVICE)
cf_ths = torch.as_tensor(rnd_cf_data[:, 6:12], dtype=torch.float32, device=DEVICE)
del rnd_cf_data

for i in tqdm(range(NUM_EPOCHS)):
    loss_array = []
    loss_weights = []
    rnd_indices = torch.randperm(data_size, dtype=torch.long, device=DEVICE)
    permuted_obs_tensors = obs_tensors[rnd_indices]
    permuted_context_tensors = context_tensors[rnd_indices]
    permuted_rtts = rtts[rnd_indices]
    for start_idx, end_idx in zip(start_indices, end_indices):
        optimizer.zero_grad()
        th_given_a_dist = TransformedDistribution(
            u_given_rtt_dist.condition(permuted_rtts[start_idx:end_idx]),
            [cs_transform_t.condition(permuted_context_tensors[start_idx:end_idx]), affine_transform])
        ln_th_given_a = th_given_a_dist.log_prob(permuted_obs_tensors[start_idx:end_idx])
        loss = -torch.mean(ln_th_given_a)
        loss.backward()
        loss_array.append(loss.detach().cpu().item())
        loss_weights.append(end_idx - start_idx)
        optimizer.step()
        u_given_rtt_dist.clear_cache()
    writer.add_scalar('nll', np.average(loss_array, weights=loss_weights), i)
    if i % 10 == 0:
        mse_list = []
        with torch.no_grad():
            latents = affine_transform.inv(obs_tensors)
            latents = cs_transform_t.condition(context_tensors).inv(latents)
            for j in range(6):
                cf_context_tensors = torch.cat([cf_actions[:, j:j+1], rtts], dim=1)
                cf_obs = cs_transform_t.condition(cf_context_tensors)(latents)
                cf_obs = affine_transform(cf_obs)
                assert cf_obs.shape == cf_ths[:, j:j+1].shape
                mse_list.append(torch.mean(torch.square(cf_obs - cf_ths[:, j:j+1])).cpu().item())
        final_mse = np.mean(mse_list)
        r2 = 1 - (final_mse / base_mse)
        writer.add_scalar('r2', r2, i)
os.makedirs(f'{args.dir}markovian/models', exist_ok=True)
torch.save(hypernet_t, f'{args.dir}markovian/models/{model_spec}_Thypernet_{NUM_EPOCHS}.pth')
torch.save(hypernet_u, f'{args.dir}markovian/models/{model_spec}_Uhypernet_{NUM_EPOCHS}.pth')