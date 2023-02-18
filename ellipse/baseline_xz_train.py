import os
import pyro.distributions as dist
import pyro.distributions.transforms as T
import torch
import pickle
import argparse
import numpy as np
from pyro.nn.dense_nn import DenseNN
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="root directory")
parser.add_argument("--seed", type=int, help="training seed", default=2)
parser.add_argument("--hw", type=int, help="width of hyper-network's hidden layers", default=64)
parser.add_argument("--n_bins", type=int, help="number of spline bins", default=16)
parser.add_argument("--bound", type=float, help="nf bound", default=3.0)
parser.add_argument("--batch_size", type=int, help="log batch size", default=20)
parser.add_argument("--dataset_seed", type=int, help="dataset random seed", default=0)
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
DEVICE = torch.device('cuda:0')
BATCH_SIZE = 2 ** args.batch_size
NUM_EPOCHS = 100
with open(f'{args.dir}dataset_{args.dataset_seed}.pkl', 'rb') as f:
    dataset = pickle.load(f)
obs_dist_base = dist.Normal(torch.zeros(3, dtype=torch.float32, device=DEVICE),
                            torch.ones(3, dtype=torch.float32, device=DEVICE))
param_dims = [args.n_bins, args.n_bins, args.n_bins - 1, args.n_bins]
param_dims2 = [2 * x for x in param_dims]
hypernets_t = [
    DenseNN(input_dim=2, hidden_dims=[args.hw, args.hw],
            param_dims=param_dims2, nonlinearity=nn.ReLU()).to(device=DEVICE),
    DenseNN(input_dim=3, hidden_dims=[args.hw, args.hw],
            param_dims=param_dims, nonlinearity=nn.ReLU()).to(device=DEVICE),
    DenseNN(input_dim=3, hidden_dims=[args.hw, args.hw],
            param_dims=param_dims, nonlinearity=nn.ReLU()).to(device=DEVICE),
    DenseNN(input_dim=2, hidden_dims=[args.hw, args.hw],
            param_dims=param_dims2, nonlinearity=nn.ReLU()).to(device=DEVICE),
    DenseNN(input_dim=3, hidden_dims=[args.hw, args.hw],
            param_dims=param_dims, nonlinearity=nn.ReLU()).to(device=DEVICE),
    DenseNN(input_dim=3, hidden_dims=[args.hw, args.hw],
            param_dims=param_dims, nonlinearity=nn.ReLU()).to(device=DEVICE)
]
modules = torch.nn.ModuleList(hypernets_t)
optimizer = torch.optim.Adam(modules.parameters())
model_spec = f'baseline_xz_bound{args.bound}_bins{args.n_bins}_hw{args.hw}_seed{args.seed}_dataset{args.dataset_seed}'
writer = SummaryWriter(log_dir=f"{args.dir}logs/{model_spec}")
model_dir = f'{args.dir}models'
os.makedirs(model_dir, exist_ok=True)
obs_splines = [
    T.SplineCoupling(input_dim=4, split_dim=2, hypernet=hypernets_t[0],
                     count_bins=args.n_bins, bound=args.bound, order="linear", identity=True),
    T.SplineCoupling(input_dim=4, split_dim=3, hypernet=hypernets_t[1],
                     count_bins=args.n_bins, bound=args.bound, order="linear", identity=True),
    T.SplineCoupling(input_dim=4, split_dim=3, hypernet=hypernets_t[2],
                     count_bins=args.n_bins, bound=args.bound, order="linear", identity=True),
    T.SplineCoupling(input_dim=4, split_dim=2, hypernet=hypernets_t[3],
                     count_bins=args.n_bins, bound=args.bound, order="linear", identity=True),
    T.SplineCoupling(input_dim=4, split_dim=3, hypernet=hypernets_t[4],
                     count_bins=args.n_bins, bound=args.bound, order="linear", identity=True),
    T.SplineCoupling(input_dim=4, split_dim=3, hypernet=hypernets_t[5],
                     count_bins=args.n_bins, bound=args.bound, order="linear", identity=True)
]
permute_obs1 = T.Permute(permutation=torch.as_tensor([0, 1, 3, 2], dtype=torch.long, device=DEVICE))
permute_obs2 = T.Permute(permutation=torch.as_tensor([0, 1, 3, 2], dtype=torch.long, device=DEVICE))
affine_transform = T.AffineTransform(
    loc=torch.as_tensor([0.0, 0.0, dataset['locs']['x'], dataset['locs']['y']], dtype=torch.float32, device=DEVICE),
    scale=torch.as_tensor([1.0, 1.0, dataset['scales']['x'], dataset['scales']['y']], dtype=torch.float32, device=DEVICE),
    event_dim=1
)
x = torch.as_tensor(dataset['data']['x'], dtype=torch.float32, device=DEVICE).unsqueeze(1)
y = torch.as_tensor(dataset['data']['y'], dtype=torch.float32, device=DEVICE).unsqueeze(1)
t = torch.as_tensor((dataset['data']['t'] - dataset['locs']['t']) / dataset['scales']['t'],
                    dtype=torch.float32, device=DEVICE).unsqueeze(1)
z = torch.as_tensor((dataset['data']['z'] - dataset['locs']['z']) / dataset['scales']['z'],
                    dtype=torch.float32, device=DEVICE).unsqueeze(1)
start_indices = np.arange(0, dataset['num_samples'], BATCH_SIZE)
end_indices = np.append(start_indices[1:], [dataset['num_samples']])
obs_transforms = [obs_splines[0], obs_splines[1], permute_obs1, obs_splines[2],
                  obs_splines[3], obs_splines[4], permute_obs2, obs_splines[5], affine_transform]
l1 = nn.L1Loss()
nlls = {}
l1s = {}
for i in tqdm(range(NUM_EPOCHS)):
    ######################## Train ########################
    loss_array = []
    loss_weights = []
    rnd_indices = torch.randperm(dataset['num_samples'], dtype=torch.long, device=DEVICE)
    xp = x[rnd_indices]
    yp = y[rnd_indices]
    zp = z[rnd_indices]
    tp = t[rnd_indices]
    for start_idx, end_idx in zip(start_indices, end_indices):
        optimizer.zero_grad()
        obs = torch.cat([tp[start_idx:end_idx], zp[start_idx:end_idx], xp[start_idx:end_idx], yp[start_idx:end_idx]], dim=1)
        ln_obs = torch.zeros((end_idx - start_idx), dtype=torch.float32, device=DEVICE)
        for transform in obs_transforms[::-1]:
            u = transform.inv(obs)
            ln_obs -= transform.log_abs_det_jacobian(u, obs)
            obs = u
        u = obs[:, 1:]
        ln_obs += torch.sum(obs_dist_base.log_prob(u), dim=1)
        loss = -torch.mean(ln_obs)
        loss.backward()
        loss_array.append(loss.detach().cpu().item())
        loss_weights.append(end_idx - start_idx)
        optimizer.step()
    writer.add_scalar('loss/nll', np.average(loss_array, weights=loss_weights), i + 1)
    nlls[i + 1] = np.average(loss_array, weights=loss_weights)
    #################### CF Error ############################
    if (i + 1) % 10 == 0 or i == 0:
        l1_array = []
        x_l1_array = []
        y_l1_array = []
        l1_weights = []
        rnd_indices = torch.randperm(dataset['num_samples'], dtype=torch.long, device=DEVICE)
        cf_t = t[rnd_indices]
        for start_idx, end_idx in zip(start_indices, end_indices):
            cf_x = torch.div(torch.cos(cf_t[start_idx:end_idx] * dataset['scales']['t'] + dataset['locs']['t']),
                             torch.cos(t[start_idx:end_idx] * dataset['scales']['t'] + dataset['locs']['t']))
            cf_x = torch.mul(cf_x, x[start_idx:end_idx])
            cf_y = torch.div(torch.sin(cf_t[start_idx:end_idx] * dataset['scales']['t'] + dataset['locs']['t']),
                             torch.sin(t[start_idx:end_idx] * dataset['scales']['t'] + dataset['locs']['t']))
            cf_y = torch.mul(cf_y, y[start_idx:end_idx])
            cf_obs = torch.cat([cf_x, cf_y], dim=1)
            with torch.no_grad():
                obs = torch.cat([t[start_idx:end_idx], z[start_idx:end_idx], x[start_idx:end_idx], y[start_idx:end_idx]], dim=1)
                for transform in obs_transforms[::-1]:
                    u = transform.inv(obs)
                    obs = u
                obs = torch.cat([cf_t[start_idx:end_idx], obs[:, 1:]], dim=1)
                for transform in obs_transforms:
                    obs = transform(obs)
                pred_x = obs[:, 2:3]
                pred_y = obs[:, 3:4]
                pred_obs = obs[:, 2:]
                assert pred_obs.shape == cf_obs.shape, f'{pred_obs.shape}, {cf_obs.shape}'
                l1_array.append(np.mean(100 * np.divide(l1(pred_obs, cf_obs).cpu().numpy(), np.abs(cf_obs.cpu().numpy()))))
                assert pred_x.shape == cf_x.shape, f'{pred_x.shape}, {cf_x.shape}'
                x_l1_array.append(np.mean(100 * np.divide(l1(pred_x, cf_x).cpu().numpy(), np.abs(cf_x.cpu().numpy()))))
                assert pred_y.shape == cf_y.shape, f'{pred_y.shape}, {cf_y.shape}'
                y_l1_array.append(np.mean(100 * np.divide(l1(pred_y, cf_y).cpu().numpy(), np.abs(cf_y.cpu().numpy()))))
                l1_weights.append(end_idx - start_idx)
        writer.add_scalar('loss/MAPE', np.average(l1_array, weights=l1_weights), i + 1)
        writer.add_scalar('loss/x_MAPE', np.average(x_l1_array, weights=l1_weights), i + 1)
        writer.add_scalar('loss/y_MAPE', np.average(y_l1_array, weights=l1_weights), i + 1)
        l1s[i + 1] = np.average(l1_array, weights=l1_weights)
    #################### Save Checkpoints ############################
    if (i + 1) % 10 == 0:
        for j, hypernet in enumerate(hypernets_t):
            torch.save(hypernet, f'{model_dir}/{model_spec}_Thypernet{j}_{i + 1}.pth')
with open(f'{model_dir}/{model_spec}_nlls.pkl', 'wb') as f:
    pickle.dump(nlls, f)
with open(f'{model_dir}/{model_spec}_l1s.pkl', 'wb') as f:
    pickle.dump(l1s, f)