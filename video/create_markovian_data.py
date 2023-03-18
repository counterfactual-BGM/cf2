import os
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="root directory")
args = parser.parse_args()
random_data = np.load(f'{args.dir}simulator_data/rnd_traj.npy')
rtts = np.load(f'{args.dir}simulator_data/rtts.npy')

# Generate training data
rnd_training_data = np.zeros((5000 * 489, 3))
index = 0
for i in range(random_data.shape[0]):
    for j in range(1, random_data.shape[1]):
        past_action_index = int(random_data[i, j, 12])
        past_th = random_data[i, j, 4]
        all_past_ths = random_data[i, j - 1, 43:49]
        all_past_sizes = random_data[i, j - 1, 13:19]
        past_size = all_past_sizes[past_action_index]
        assert past_th == all_past_ths[past_action_index]
        rnd_training_data[index, 0] = past_size
        rnd_training_data[index, 1] = past_th
        rnd_training_data[index, 2] = rtts[i]
        index = index + 1

size_mean, size_std = np.mean(rnd_training_data[:, 0]), np.std(rnd_training_data[:, 0])
th_mean, th_std = np.mean(rnd_training_data[:, 1]), np.std(rnd_training_data[:, 1])
rtt_mean, rtt_std = np.mean(rtts), np.std(rtts)
rnd_locs = np.array([size_mean, th_mean + 1.3 * th_std, rtt_mean])
rnd_scales = np.array([size_std, th_std, rtt_std])
os.makedirs(f'{args.dir}markovian', exist_ok=True)
np.save(f'{args.dir}markovian/training_data.npy', rnd_training_data)
np.save(f'{args.dir}markovian/locs.npy', rnd_locs)
np.save(f'{args.dir}markovian/scales.npy', rnd_scales)

# Generate ground-truth counterfactuals
rnd_cf_data = np.zeros((5000 * 489, 12))
index = 0
squared_error_list = []
for i in tqdm(range(random_data.shape[0])):
    for j in range(1, random_data.shape[1]):
        past_action_index = int(random_data[i, j, 12])
        all_past_ths = random_data[i, j - 1, 43:49]
        all_past_sizes = random_data[i, j - 1, 13:19]
        past_th = random_data[i, j, 4]
        past_size = all_past_sizes[past_action_index]
        squared_error = (all_past_ths - past_th) ** 2
        squared_error_list.extend(squared_error)
        rnd_cf_data[index, 0:6] = all_past_sizes
        rnd_cf_data[index, 6:12] = all_past_ths
        index = index + 1
base_mse = np.mean(squared_error_list)
np.save(f'{args.dir}markovian/cf_data.npy', rnd_cf_data)
np.save(f'{args.dir}markovian/cf_base_mse.npy', base_mse)
