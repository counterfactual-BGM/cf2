import os
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="root directory")
args = parser.parse_args()

# Generate training data
algo_names = ['bba', 'bola', 'mpc', 'opt_rate', 'pess_rate', 'rate', 'rnd', 'bbamix_X2.0_RND50%', 'bbamix_X1.0_RND50%']
all_ths = []
all_sizes = []
data = {algo: None for algo in algo_names}
for algo_idx, algo in tqdm(enumerate(algo_names)):
    algo_data = np.load(f'{args.dir}simulator_data/{algo}_traj.npy')
    data[algo] = algo_data
    for i in range(algo_data.shape[0]):
        for j in range(1, algo_data.shape[1]):
            past_action_index = int(algo_data[i, j, 12])
            past_th = algo_data[i, j, 4]
            all_past_sizes = algo_data[i, j-1, 13:19]
            past_size = all_past_sizes[past_action_index]
            all_ths.append(past_th)
            all_sizes.append(past_size)
th_mean, th_std = np.mean(all_ths), np.std(all_ths)
size_mean, size_std = np.mean(all_sizes), np.std(all_sizes)
rtts = np.load(f'{args.dir}simulator_data/rtts.npy')
rtt_mean, rtt_std = np.mean(rtts), np.std(rtts)
locs = np.array([size_mean + 1.4 * size_std, th_mean + 1.6* th_std, rtt_mean])
scales = np.array([size_std, th_std, rtt_std])
algo_dir = f'{args.dir}iv'
os.makedirs(algo_dir, exist_ok=True)
index = 0
algo_training_data = np.zeros((5000*489*len(algo_names), 4))
for algo_idx, algo in enumerate(algo_names):
    algo_data = data[algo]
    for i in range(algo_data.shape[0]):
        for j in range(1, algo_data.shape[1]):
            past_action_index = int(algo_data[i, j, 12])
            past_th = algo_data[i, j, 4]
            all_past_ths = algo_data[i, j-1, 43:49]
            all_past_sizes = algo_data[i, j-1, 13:19]
            past_size = all_past_sizes[past_action_index]
            assert past_th == all_past_ths[past_action_index]
            algo_training_data[index, 0] = past_size
            algo_training_data[index, 1] = past_th
            algo_training_data[index, 2] = rtts[i]
            algo_training_data[index, 3] = algo_idx
            index = index + 1
np.save(f'{algo_dir}/training_data.npy', algo_training_data)
np.save(f'{algo_dir}/locs.npy', locs)
np.save(f'{algo_dir}/scales.npy', scales)

# Generate ground-truth counterfactuals
cf_data = np.zeros((5000 * 489, 12))
index = 0
squared_error_list = []
random_data = np.load(f'{args.dir}simulator_data/rnd_traj.npy')
for i in tqdm(range(random_data.shape[0])):
    for j in range(1, random_data.shape[1]):
        past_action_index = int(random_data[i, j, 12])
        all_past_ths = random_data[i, j - 1, 43:49]
        all_past_sizes = random_data[i, j - 1, 13:19]
        past_th = random_data[i, j, 4]
        past_size = all_past_sizes[past_action_index]
        squared_error = (all_past_ths - past_th) ** 2
        squared_error_list.extend(squared_error)
        cf_data[index, 0:6] = all_past_sizes
        cf_data[index, 6:12] = all_past_ths
        index = index + 1

mse_dic = {algo: None for algo in algo_names}
for algo_idx, algo in tqdm(enumerate(algo_names)):
    squared_errors = []
    algo_data = data[algo]
    index = 0
    for i in range(algo_data.shape[0]):
        for j in range(1, algo_data.shape[1]):
            past_th = algo_data[i, j, 4]
            squared_errors.append((cf_data[index, 6:12] - past_th) ** 2)
            index = index + 1
    mse = np.mean(squared_errors)
    mse_dic[algo] = mse

base_mse = np.mean([mse_dic[algo] for algo in algo_names])
np.save(f'{algo_dir}/cf_base_mse.npy', base_mse)
np.save(f'{args.dir}/cf_data.npy', cf_data)