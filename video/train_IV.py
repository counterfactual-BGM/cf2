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

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, help="seed")
parser.add_argument("--dir", help="root directory")
parser.add_argument("--batch_size", type=int, help="log batch size", default=20)
parser.add_argument("--device", type=str, help="Compute device", default='cuda:0')
parser.add_argument("--start_epoch", type=int, default=0, help="epoch number to start with")
parser.add_argument("--hw", type=int, help="width of hyper-network's hidden layers", default=64)
parser.add_argument("--n_bins", type=int, help="number of spline bins", default=16)
parser.add_argument('--no_val', action='store_true', help='skip validation')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
DEVICE = torch.device(args.device)
NUM_EPOCHS = 150
BATCH_SIZE = 2 ** args.batch_size
model_dir = f'{args.dir}iv/models'
os.makedirs(model_dir, exist_ok=True)

training_data = np.load(f'{args.dir}iv/training_data.npy')
locs = np.load(f'{args.dir}iv/locs.npy')
scales = np.load(f'{args.dir}iv/scales.npy')
cf_data = np.load(f'{args.dir}iv/cf_data.npy')
base_mse = np.load(f'{args.dir}iv/cf_base_mse.npy')
cf_data[:, 0:6] = (cf_data[:, 0:6] - locs[0]) / scales[0]  # size

data_size = training_data.shape[0]
start_indices = np.arange(0, data_size, BATCH_SIZE)
end_indices = np.append(start_indices[1:], [data_size])
param_dims = [args.n_bins, args.n_bins, args.n_bins - 1, args.n_bins]

model_specs = f'bins{args.n_bins}_hw{args.hw}_seed{args.seed}'
if args.start_epoch > 0:
    hypernet_t = torch.load(f'{model_dir}/{model_specs}_Thypernet_{args.start_epoch}.pth')
    hypernet_a = torch.load(f'{model_dir}/{model_specs}_Ahypernet_{args.start_epoch}.pth')
    hypernet_u = torch.load(f'{model_dir}/{model_specs}_Uhypernet_{args.start_epoch}.pth')
else:
    hypernet_a = DenseNN(
        input_dim=11, hidden_dims=[args.hw, args.hw], param_dims=param_dims, nonlinearity=nn.ReLU()).to(device=DEVICE)
    hypernet_t = DenseNN(
        input_dim=2, hidden_dims=[args.hw, args.hw], param_dims=param_dims, nonlinearity=nn.ReLU()).to(device=DEVICE)
    hypernet_u = DenseNN(
        input_dim=1, hidden_dims=[args.hw, args.hw], param_dims=param_dims, nonlinearity=nn.ReLU()).to(device=DEVICE)
modules = torch.nn.ModuleList([hypernet_a, hypernet_t, hypernet_u])
optimizer = torch.optim.Adam(modules.parameters())
writer = SummaryWriter(log_dir=f"{args.dir}iv/logs/{model_specs}")
u_dist_base = dist.Normal(torch.zeros(1, dtype=torch.float32, device=DEVICE),
                          torch.ones(1, dtype=torch.float32, device=DEVICE))
a_dist_base = dist.Normal(torch.zeros(1, dtype=torch.float32, device=DEVICE),
                          torch.ones(1, dtype=torch.float32, device=DEVICE))
cs_transform_a = T.ConditionalSpline(nn=hypernet_a, input_dim=1, count_bins=args.n_bins, bound=3.5, order='linear')
cs_transform_t = T.ConditionalSpline(nn=hypernet_t, input_dim=1, count_bins=args.n_bins, bound=3.5, order='linear')
cs_transform_u = T.ConditionalSpline(nn=hypernet_u, input_dim=1, count_bins=args.n_bins, bound=3.5, order='linear')
affine_transform = T.AffineTransform(loc=locs[1], scale=scales[1])  # Throughput
actions = torch.as_tensor((training_data[:, 0:1] - locs[0]) / scales[0], dtype=torch.float32, device=DEVICE)
rtts = torch.as_tensor((training_data[:, 2:3] - locs[2]) / scales[2], dtype=torch.float32, device=DEVICE)
algo_indices = torch.as_tensor(training_data[:, 3], dtype=torch.long, device=DEVICE)
algo_one_hot = nn.functional.one_hot(algo_indices)
del algo_indices
obs_tensors = torch.as_tensor(training_data[:, 1:2], dtype=torch.float32, device=DEVICE)
t_context_tensors = torch.cat([actions, rtts], dim=1)

cf_actions = torch.as_tensor(cf_data[:, 0:6], dtype=torch.float32, device=DEVICE)
cf_ths = torch.as_tensor(cf_data[:, 6:12], dtype=torch.float32, device=DEVICE)

cf_actions = torch.cat([cf_actions for _ in range(9)], dim=0)
cf_ths = torch.cat([cf_ths for _ in range(9)], dim=0)
del cf_data
a_given_u_and_algo_dist = dist.ConditionalTransformedDistribution(a_dist_base, [cs_transform_a])
u_given_rtt_dist = dist.ConditionalTransformedDistribution(u_dist_base,)
for i in tqdm(range(args.start_epoch, NUM_EPOCHS)):
    loss_array = []
    t_array = []
    a_array = []
    loss_weights = []
    rnd_indices = torch.randperm(data_size, dtype=torch.long, device=DEVICE)
    permuted_obs_tensors = obs_tensors[rnd_indices]
    permuted_t_context_tensors = t_context_tensors[rnd_indices]
    permuted_actions = actions[rnd_indices]
    permuted_algos = algo_one_hot[rnd_indices]
    permuted_rtts = rtts[rnd_indices]
    for start_idx, end_idx in zip(start_indices, end_indices):
        optimizer.zero_grad()
        conditioned_cs_transform_t = cs_transform_t.condition(permuted_t_context_tensors[start_idx:end_idx])
        half_u = affine_transform.inv(permuted_obs_tensors[start_idx:end_idx])
        ln_t_given_a = - affine_transform.log_abs_det_jacobian(half_u, permuted_obs_tensors[start_idx:end_idx])
        u = conditioned_cs_transform_t.inv(half_u)
        ln_t_given_a = ln_t_given_a - conditioned_cs_transform_t.log_abs_det_jacobian(u, half_u)
        conditioned_u_given_rtt = u_given_rtt_dist.condition(permuted_rtts[start_idx:end_idx])
        ln_t_given_a += conditioned_u_given_rtt.log_prob(u)
        ln_t_given_a = torch.mean(ln_t_given_a)
        a_condition = torch.cat([u, permuted_rtts[start_idx:end_idx], permuted_algos[start_idx:end_idx]], dim=1)
        conditioned_a_given_u_and_algo = a_given_u_and_algo_dist.condition(a_condition)
        ln_a_given_u_and_algo = torch.mean(conditioned_a_given_u_and_algo.log_prob(permuted_actions[start_idx:end_idx]))
        loss = - (ln_t_given_a + ln_a_given_u_and_algo)
        loss.backward()
        loss_array.append(loss.detach().cpu().item())
        t_array.append((-ln_t_given_a).detach().cpu().item())
        a_array.append((-ln_a_given_u_and_algo).detach().cpu().item())
        loss_weights.append(end_idx - start_idx)
        optimizer.step()
        a_given_u_and_algo_dist.clear_cache()
        u_given_rtt_dist.clear_cache()
    writer.add_scalar('nll', np.average(t_array, weights=loss_weights), i)
    writer.add_scalar('a_nll', np.average(a_array, weights=loss_weights), i)
    writer.add_scalar('total_nll', np.average(loss_array, weights=loss_weights), i)
    ##############################################################################
    if args.no_val:
        pass
    else:
        if (i+1) % 10 == 0:
            mse_list = []
            mse_weights = []
            for start_idx, end_idx in zip(start_indices, end_indices):
                with torch.no_grad():
                    latents = affine_transform.inv(obs_tensors[start_idx:end_idx])
                    latents = cs_transform_t.condition(t_context_tensors[start_idx:end_idx]).inv(latents)
                    for j in range(6):
                        cf_context_tensors = torch.cat(
                            [cf_actions[start_idx:end_idx, j:j + 1], rtts[start_idx:end_idx]], dim=1)
                        cf_obs = latents
                        cf_obs = cs_transform_t.condition(cf_context_tensors)(cf_obs)
                        cf_obs = affine_transform(cf_obs)
                        assert cf_obs.shape == cf_ths[start_idx:end_idx, j:j + 1].shape
                        mse_list.append(
                            torch.mean(torch.square(cf_obs - cf_ths[start_idx:end_idx, j:j + 1])).cpu().item())
                        mse_weights.append(end_idx - start_idx)
            final_mse = np.average(mse_list, weights=mse_weights)
            r2 = 1 - (final_mse / base_mse)
            writer.add_scalar('r2', r2, i + 1)
torch.save(hypernet_t, f'{model_dir}/{model_specs}_Thypernet_{NUM_EPOCHS}.pth')
torch.save(hypernet_a, f'{model_dir}/{model_specs}_Ahypernet_{NUM_EPOCHS}.pth')
torch.save(hypernet_u, f'{model_dir}/{model_specs}_Uhypernet_{NUM_EPOCHS}.pth')
