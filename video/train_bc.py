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
parser.add_argument("--algo", help="buffer based algorithm")
parser.add_argument("--seed", type=int, help="seed")
parser.add_argument("--batch_size", type=int, help="log batch size", default=20)
parser.add_argument("--n_bins", type=int, help="number of spline bins", default=16)
parser.add_argument("--hw", type=int, help="width of hyper-network's hidden layers", default=64)
parser.add_argument("--device", type=str, help="Compute device", default='cuda:0')
args = parser.parse_args()
DEVICE = torch.device(args.device)
NUM_EPOCHS = 300
BATCH_SIZE = 2 ** args.batch_size
model_spec = f'bins{args.n_bins}_hw{args.hw}_seed{args.seed}'

training_data = np.load(f'{args.dir}bc/{args.algo}/training_data.npy')
locs = np.load(f'{args.dir}bc/{args.algo}/locs.npy')
scales = np.load(f'{args.dir}bc/{args.algo}/scales.npy')
cf_data = np.load(f'{args.dir}bc/{args.algo}/cf_data.npy')
base_mse = np.load(f'{args.dir}bc/{args.algo}/cf_base_mse.npy')
cf_data[:, 0:6] = (cf_data[:, 0:6] - locs[0]) / scales[0]

data_size = training_data.shape[0]
start_indices = np.arange(0, data_size, BATCH_SIZE)
end_indices = np.append(start_indices[1:], [data_size])
param_dims = [args.n_bins, args.n_bins, args.n_bins - 1, args.n_bins]
hypernet_u = DenseNN(
    input_dim=2, hidden_dims=[args.hw, args.hw], param_dims=param_dims, nonlinearity=nn.ReLU()).to(device=DEVICE)
hypernet_t = DenseNN(
    input_dim=2, hidden_dims=[args.hw, args.hw], param_dims=param_dims, nonlinearity=nn.ReLU()).to(device=DEVICE)
modules = torch.nn.ModuleList([hypernet_t, hypernet_u])
optimizer = torch.optim.Adam(modules.parameters())
writer = SummaryWriter(log_dir=f"{args.dir}{args.algo}/logs/{model_spec}")
dist_base = dist.Normal(torch.zeros(1, dtype=torch.float32, device=DEVICE),
                        torch.ones(1, dtype=torch.float32, device=DEVICE))
cs_transform_u = T.ConditionalSpline(nn=hypernet_u, input_dim=1, count_bins=args.n_bins, bound=3.5, order='linear')
cs_transform_t = T.ConditionalSpline(nn=hypernet_t, input_dim=1, count_bins=args.n_bins, bound=3.5, order='linear')
affine_transform = T.AffineTransform(loc=locs[1], scale=scales[1])
actions = torch.as_tensor((training_data[:, 0:1] - locs[0]) / scales[0], dtype=torch.float32, device=DEVICE)
rtts = torch.as_tensor((training_data[:, 2:3] - locs[2]) / scales[2], dtype=torch.float32, device=DEVICE)
buffs = torch.as_tensor((training_data[:, 3:4] - locs[3]) / scales[3], dtype=torch.float32, device=DEVICE)

obs_tensors = torch.as_tensor(training_data[:, 1:2], dtype=torch.float32, device=DEVICE)
t_context_tensors = torch.cat([actions, rtts], dim=1)

cf_actions = torch.as_tensor(cf_data[:, 0:6], dtype=torch.float32, device=DEVICE)
cf_ths = torch.as_tensor(cf_data[:, 6:12], dtype=torch.float32, device=DEVICE)
del cf_data

u_given_b_dist = dist.ConditionalTransformedDistribution(dist_base, [cs_transform_u])

for i in tqdm(range(NUM_EPOCHS)):
    loss_array = []
    loss_weights = []
    rnd_indices = torch.randperm(data_size, dtype=torch.long, device=DEVICE)
    permuted_obs_tensors = obs_tensors[rnd_indices]
    permuted_t_context_tensors = t_context_tensors[rnd_indices]
    permuted_buffs = buffs[rnd_indices]
    permuted_rtts = rtts[rnd_indices]
    for start_idx, end_idx in zip(start_indices, end_indices):
        optimizer.zero_grad()
        th_given_a_dist = TransformedDistribution(
            u_given_b_dist.condition(torch.cat([permuted_buffs[start_idx:end_idx], permuted_rtts[start_idx:end_idx]],
                                               dim=1)),
            [cs_transform_t.condition(permuted_t_context_tensors[start_idx:end_idx]), affine_transform])
        ln_th_given_a = th_given_a_dist.log_prob(permuted_obs_tensors[start_idx:end_idx])
        loss = -torch.mean(ln_th_given_a)
        loss.backward()
        loss_array.append(loss.detach().cpu().item())
        loss_weights.append(end_idx - start_idx)
        optimizer.step()
        u_given_b_dist.clear_cache()
    writer.add_scalar('nll', np.average(loss_array, weights=loss_weights), i)
    if (i + 1) % 10 == 0:
        mse_list = []
        with torch.no_grad():
            latents = affine_transform.inv(obs_tensors)
            latents = cs_transform_t.condition(t_context_tensors).inv(latents)
            for j in range(6):
                cf_context_tensors = torch.cat([cf_actions[:, j:j+1], rtts], dim=1)
                cf_obs = cs_transform_t.condition(cf_context_tensors)(latents)
                cf_obs = affine_transform(cf_obs)
                assert cf_obs.shape == cf_ths[:, j:j+1].shape
                mse_list.append(torch.mean(torch.square(cf_obs - cf_ths[:, j:j+1])).cpu().item())
        final_mse = np.mean(mse_list)
        r2 = 1 - (final_mse / base_mse)
        writer.add_scalar('r2', r2, i + 1)
model_dir = f'{args.dir}bc/{args.algo}/models'
os.makedirs(model_dir, exist_ok=True)
torch.save(hypernet_t, f'{model_dir}/{model_spec}_Thypernet_{NUM_EPOCHS}.pth')
torch.save(hypernet_u, f'{model_dir}/{model_spec}_Uhypernet_{NUM_EPOCHS}.pth')
