import pyro.distributions.transforms as T
import torch
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
e_t = dataset['data']['t'][658896]
e_b = dataset['data']['b'][658896]
e_r = dataset['data']['r'][658896]
e_z = dataset['data']['z'][658896]
e_x = dataset['data']['x'][658896]
e_y = dataset['data']['y'][658896]
anchor = {'x': e_x, 'y': e_y}
with open(f'{args.dir}anchor.pkl', 'wb') as f:
    pickle.dump(anchor, f)
num_samples = 1000000

##################################################### BC (OURS) #####################################################
model_spec = f'bound{args.bound}_bins{args.n_bins}_hw{args.hw}_seed{args.seed}_dataset{args.dataset_seed}'
model_dir = f'{args.dir}models'
hypernets_u = [torch.load(f'{model_dir}/{model_spec}_Uhypernet{j}_{NUM_EPOCHS}.pth') for j in range(3)]
hypernets_t = [torch.load(f'{model_dir}/{model_spec}_Thypernet{j}_{NUM_EPOCHS}.pth') for j in range(3)]
obs_splines = [
    T.SplineCoupling(input_dim=3, split_dim=1, hypernet=hypernets_t[0],
                     count_bins=args.n_bins, bound=args.bound, order="linear", identity=True),
    T.SplineCoupling(input_dim=3, split_dim=2, hypernet=hypernets_t[1],
                     count_bins=args.n_bins, bound=args.bound, order="linear", identity=True),
    T.SplineCoupling(input_dim=3, split_dim=2, hypernet=hypernets_t[2],
                     count_bins=args.n_bins, bound=args.bound, order="linear", identity=True)
]
permute_obs = T.Permute(permutation=torch.as_tensor([0, 2, 1], dtype=torch.long, device=DEVICE))
affine_transform = T.AffineTransform(
    loc=torch.as_tensor([0.0, dataset['locs']['x'], dataset['locs']['y']], dtype=torch.float32, device=DEVICE),
    scale=torch.as_tensor([1.0, dataset['scales']['x'], dataset['scales']['y']], dtype=torch.float32, device=DEVICE),
    event_dim=1
)
obs_transforms = [obs_splines[0], obs_splines[1], permute_obs, obs_splines[2], affine_transform]

cf_t = torch.rand((num_samples, 1), dtype=torch.float32, device=DEVICE) * 2 * np.pi
normalized_cf_t = (cf_t - dataset['locs']['t']) / dataset['scales']['t']
t = torch.ones((num_samples, 1), dtype=torch.float32, device=DEVICE) * (e_t - dataset['locs']['t']) / dataset['scales'][
    't']
x = torch.ones((num_samples, 1), dtype=torch.float32, device=DEVICE) * e_x
y = torch.ones((num_samples, 1), dtype=torch.float32, device=DEVICE) * e_y
obs = torch.cat([t, x, y], dim=1)
for transform in obs_transforms[::-1]:
    with torch.no_grad():
        u = transform.inv(obs)
        obs = u
obs = torch.cat([normalized_cf_t, obs[:, 1:]], dim=1)
for transform in obs_transforms:
    with torch.no_grad():
        obs = transform(obs)
pred_x = obs[:, 1:2]
pred_y = obs[:, 2:3]
pred_obs = obs[:, 1:]
cf_t = cf_t[:, 0].cpu().numpy()
gt_x = e_r * e_b * (2 + np.cos(cf_t))
gt_y = e_b * (2 + np.sin(cf_t))
bc = {'x': pred_x[:, 0].cpu().numpy(), 'y': pred_y[:, 0].cpu().numpy()}
gt = {'x': gt_x, 'y': gt_y}
with open(f'{args.dir}bc.pkl', 'wb') as f:
    pickle.dump(bc, f)
with open(f'{args.dir}gt.pkl', 'wb') as f:
    pickle.dump(gt, f)

##################################################### Baseline-x #####################################################

param_dims = [args.n_bins, args.n_bins, args.n_bins - 1, args.n_bins]
param_dims2 = [2 * x for x in param_dims]
model_spec = f'baseline_x_bound{args.bound}_bins{args.n_bins}_hw{args.hw}_seed{args.seed}_dataset{args.dataset_seed}'
model_dir = f'{args.dir}models'
hypernets_t = [torch.load(f'{model_dir}/{model_spec}_Thypernet{j}_{NUM_EPOCHS}.pth') for j in range(6)]
obs_splines = [
    T.SplineCoupling(input_dim=3, split_dim=1, hypernet=hypernets_t[0],
                     count_bins=args.n_bins, bound=args.bound, order="linear", identity=True),
    T.SplineCoupling(input_dim=3, split_dim=2, hypernet=hypernets_t[1],
                     count_bins=args.n_bins, bound=args.bound, order="linear", identity=True),
    T.SplineCoupling(input_dim=3, split_dim=2, hypernet=hypernets_t[2],
                     count_bins=args.n_bins, bound=args.bound, order="linear", identity=True),
    T.SplineCoupling(input_dim=3, split_dim=1, hypernet=hypernets_t[3],
                     count_bins=args.n_bins, bound=args.bound, order="linear", identity=True),
    T.SplineCoupling(input_dim=3, split_dim=2, hypernet=hypernets_t[4],
                     count_bins=args.n_bins, bound=args.bound, order="linear", identity=True),
    T.SplineCoupling(input_dim=3, split_dim=2, hypernet=hypernets_t[5],
                     count_bins=args.n_bins, bound=args.bound, order="linear", identity=True)
]
permute_obs1 = T.Permute(permutation=torch.as_tensor([0, 2, 1], dtype=torch.long, device=DEVICE))
permute_obs2 = T.Permute(permutation=torch.as_tensor([0, 2, 1], dtype=torch.long, device=DEVICE))
affine_transform = T.AffineTransform(
    loc=torch.as_tensor([0.0, dataset['locs']['x'], dataset['locs']['y']], dtype=torch.float32, device=DEVICE),
    scale=torch.as_tensor([1.0, dataset['scales']['x'], dataset['scales']['y']], dtype=torch.float32, device=DEVICE),
    event_dim=1
)
obs_transforms = [obs_splines[0], obs_splines[1], permute_obs1, obs_splines[2],
                  obs_splines[3], obs_splines[4], permute_obs2, obs_splines[5], affine_transform]
cf_t = torch.rand((num_samples, 1), dtype=torch.float32, device=DEVICE) * 2 * np.pi
normalized_cf_t = (cf_t - dataset['locs']['t']) / dataset['scales']['t']
t = torch.ones((num_samples, 1), dtype=torch.float32, device=DEVICE) * (e_t - dataset['locs']['t']) / dataset['scales']['t']
x = torch.ones((num_samples, 1), dtype=torch.float32, device=DEVICE) * e_x
y = torch.ones((num_samples, 1), dtype=torch.float32, device=DEVICE) * e_y
obs = torch.cat([t, x, y], dim=1)
for transform in obs_transforms[::-1]:
    u = transform.inv(obs)
    obs = u
obs = torch.cat([normalized_cf_t, obs[:, 1:]], dim=1)
for transform in obs_transforms:
    with torch.no_grad():
        obs = transform(obs)
pred_x = obs[:, 1:2]
pred_y = obs[:, 2:3]
pred_obs = obs[:, 1:]
cf_t = cf_t[:, 0].cpu().numpy()
base_x = {'x': pred_x[:, 0].cpu().numpy(), 'y': pred_y[:, 0].cpu().numpy()}
with open(f'{args.dir}baseline_x.pkl', 'wb') as f:
    pickle.dump(base_x, f)

##################################################### Baseline-xz #####################################################

param_dims = [args.n_bins, args.n_bins, args.n_bins - 1, args.n_bins]
param_dims2 = [2 * x for x in param_dims]
model_spec = f'baseline_xz_bound{args.bound}_bins{args.n_bins}_hw{args.hw}_seed{args.seed}_dataset{args.dataset_seed}'
model_dir = f'{args.dir}models'
hypernets_t = [torch.load(f'{model_dir}/{model_spec}_Thypernet{j}_{NUM_EPOCHS}.pth') for j in range(6)]
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
obs_transforms = [obs_splines[0], obs_splines[1], permute_obs1, obs_splines[2],
                  obs_splines[3], obs_splines[4], permute_obs2, obs_splines[5], affine_transform]
cf_t = torch.rand((num_samples, 1), dtype=torch.float32, device=DEVICE) * 2 * np.pi
normalized_cf_t = (cf_t - dataset['locs']['t']) / dataset['scales']['t']
t = torch.ones((num_samples, 1), dtype=torch.float32, device=DEVICE) * (e_t - dataset['locs']['t']) / dataset['scales']['t']
x = torch.ones((num_samples, 1), dtype=torch.float32, device=DEVICE) * e_x
y = torch.ones((num_samples, 1), dtype=torch.float32, device=DEVICE) * e_y
z = torch.ones((num_samples, 1), dtype=torch.float32, device=DEVICE) * (e_z - dataset['locs']['z']) / dataset['scales']['z']
obs = torch.cat([t, z, x, y], dim=1)
for transform in obs_transforms[::-1]:
    with torch.no_grad():
        u = transform.inv(obs)
        obs = u
obs = torch.cat([normalized_cf_t, obs[:, 1:]], dim=1)
for transform in obs_transforms:
    with torch.no_grad():
        obs = transform(obs)
pred_x = obs[:, 2:3]
pred_y = obs[:, 3:4]
base_xz = {'x': pred_x[:, 0].cpu().numpy(), 'y': pred_y[:, 0].cpu().numpy()}
with open(f'{args.dir}baseline_xz.pkl', 'wb') as f:
    pickle.dump(base_xz, f)

##################################################### Plot #####################################################

plt.rcParams.update({
    "text.usetex": True,
    'legend.fontsize': 6,              # -> legends
    'font.family': 'serif',
    'font.serif': ['Times']
})
plt.rcParams['axes.axisbelow'] = True

size = 0.0001
plt.figure(figsize=(3.25, 1.3))
plt.scatter(bc['x'], bc['y'], s=size, label='bc', color='C1')
plt.scatter(gt['x'], gt['y'], s=size, label='gt', color='C2')
plt.scatter(base_x['x'], base_x['y'], s=size, color='C3')
plt.scatter(base_xz['x'], base_xz['y'], s=size, color='C9')
plt.scatter(anchor['x'], anchor['y'], s=25, label='anchor', color='w', edgecolor='C2', marker='o',zorder=100)
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
handles = [Line2D([0], [0], color='C1'),
          Line2D([0], [0], color='C2'),
          Line2D([0], [0], color='C3'),
          Line2D([0], [0], color='C9'),
          Line2D([0], [0], color='w', markerfacecolor='w', markeredgecolor='C2', marker='o', ms=6)]
plt.legend(handles, ['BC (ours)', 'ground-truth', 'baseline-x', 'baseline-xz', 'evidence'], ncol=5, loc='lower left', bbox_to_anchor=(0, 1.1, 1, 0.25), borderaxespad=0, borderpad=0.5, mode="expand", handletextpad=0.5)
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.76)
plt.grid()
plt.savefig(f'{args.dir}ellipse', dpi=1200)