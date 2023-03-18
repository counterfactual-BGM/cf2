import os
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="root directory")
args = parser.parse_args()

# Generate training data
bba_data = np.load(f'{args.dir}simulator_data/bba_traj.npy')
bola_data = np.load(f'{args.dir}simulator_data/bola_traj.npy')
rtts = np.load(f'{args.dir}simulator_data/rtts.npy')
# BBA
bba_dir = f'{args.dir}bc/bba'
os.makedirs(bba_dir, exist_ok=True)
bba_training_data = np.zeros((5000*489, 4))
index = 0
for i in range(bba_data.shape[0]):
    for j in range(1, bba_data.shape[1]):
        past_action_index = int(bba_data[i, j, 12])
        past_th = bba_data[i, j, 4]
        past_buffer = bba_data[i, j-1, 10]
        all_past_ths = bba_data[i, j-1, 43:49]
        all_past_sizes = bba_data[i, j-1, 13:19]
        past_size = all_past_sizes[past_action_index]
        assert past_th == all_past_ths[past_action_index]
        bba_training_data[index, 0] = past_size
        bba_training_data[index, 1] = past_th
        bba_training_data[index, 2] = rtts[i]
        bba_training_data[index, 3] = past_buffer
        index = index + 1
size_mean, size_std = np.mean(bba_training_data[:, 0]), np.std(bba_training_data[:, 0])
th_mean, th_std = np.mean(bba_training_data[:, 1]), np.std(bba_training_data[:, 1])
rtt_mean, rtt_std = np.mean(rtts), np.std(rtts)
buff_mean, buff_std = np.mean(bba_training_data[:, 3]), np.std(bba_training_data[:, 3])
bba_locs = np.array([size_mean, th_mean+1.3*th_std, rtt_mean, buff_mean])
bba_scales = np.array([size_std, th_std, rtt_std, buff_std])
np.save(f'{bba_dir}/training_data.npy', bba_training_data)
np.save(f'{bba_dir}/rnd_locs.npy', bba_locs)
np.save(f'{bba_dir}/rnd_scales.npy', bba_scales)

# Bola
bola_dir = f'{args.dir}bc/bola'
os.makedirs(bola_dir, exist_ok=True)
bola_training_data = np.zeros((5000*489, 4))
index = 0
for i in range(bola_data.shape[0]):
    for j in range(1, bola_data.shape[1]):
        past_action_index = int(bola_data[i, j, 12])
        past_th = bola_data[i, j, 4]
        past_buffer = bola_data[i, j-1, 10]
        all_past_ths = bola_data[i, j-1, 43:49]
        all_past_sizes = bola_data[i, j-1, 13:19]
        past_size = all_past_sizes[past_action_index]
        assert past_th == all_past_ths[past_action_index]
        bola_training_data[index, 0] = past_size
        bola_training_data[index, 1] = past_th
        bola_training_data[index, 2] = rtts[i]
        bola_training_data[index, 3] = past_buffer
        index = index + 1
size_mean, size_std = np.mean(bola_training_data[:, 0]), np.std(bola_training_data[:, 0])
th_mean, th_std = np.mean(bola_training_data[:, 1]), np.std(bola_training_data[:, 1])
rtt_mean, rtt_std = np.mean(rtts), np.std(rtts)
buff_mean, buff_std = np.mean(bola_training_data[:, 3]), np.std(bola_training_data[:, 3])
bola_locs = np.array([size_mean, th_mean+1.4*th_std, rtt_mean, buff_mean])
bola_scales = np.array([size_std, th_std, rtt_std, buff_std])
np.save(f'{bola_dir}/training_data.npy', bola_training_data)
np.save(f'{bola_dir}/locs.npy', bola_locs)
np.save(f'{bola_dir}/scales.npy', bola_scales)

# Generate ground-truth counterfactuals
# BBA
bba_cf_data = np.zeros((5000*489, 12))
index = 0
squared_error_list = []
for i in tqdm(range(bba_data.shape[0])):
    for j in range(1, bba_data.shape[1]):
        past_action_index = int(bba_data[i, j, 12])
        all_past_ths = bba_data[i, j-1, 43:49]
        all_past_sizes = bba_data[i, j-1, 13:19]
        past_th = bba_data[i, j, 4]
        past_size = all_past_sizes[past_action_index]
        squared_error = (all_past_ths - past_th) ** 2
        squared_error_list.extend(squared_error)
        bba_cf_data[index, 0:6] = all_past_sizes
        bba_cf_data[index, 6:12] = all_past_ths
        index = index + 1
bba_base_mse = np.mean(squared_error_list)
np.save(f'{bba_dir}/cf_data.npy', bba_cf_data)
np.save(f'{bba_dir}/cf_base_mse.npy', bba_base_mse)

#BOLA
bola_cf_data = np.zeros((5000*489, 12))
index = 0
squared_error_list = []
for i in tqdm(range(bola_data.shape[0])):
    for j in range(1, bola_data.shape[1]):
        past_action_index = int(bola_data[i, j, 12])
        all_past_ths = bola_data[i, j-1, 43:49]
        all_past_sizes = bola_data[i, j-1, 13:19]
        past_th = bola_data[i, j, 4]
        past_size = all_past_sizes[past_action_index]
        squared_error = (all_past_ths - past_th) ** 2
        squared_error_list.extend(squared_error)
        bola_cf_data[index, 0:6] = all_past_sizes
        bola_cf_data[index, 6:12] = all_past_ths
        index = index + 1
bola_base_mse = np.mean(squared_error_list)
np.save(f'{bola_dir}/cf_data.npy', bola_cf_data)
np.save(f'{bola_dir}/cf_base_mse.npy', bola_base_mse)

# MIX
mix_dir = f'{args.dir}mix'
mix_training_data = np.zeros((5000*489*2, 4))
index = 0
for i in range(bola_data.shape[0]):
    for j in range(1, bola_data.shape[1]):
        past_action_index = int(bola_data[i, j, 12])
        past_th = bola_data[i, j, 4]
        past_buffer = bola_data[i, j-1, 10]
        all_past_ths = bola_data[i, j-1, 43:49]
        all_past_sizes = bola_data[i, j-1, 13:19]
        past_size = all_past_sizes[past_action_index]
        assert past_th == all_past_ths[past_action_index]
        mix_training_data[index, 0] = past_size
        mix_training_data[index, 1] = past_th
        mix_training_data[index, 2] = rtts[i]
        mix_training_data[index, 3] = past_buffer
        index = index + 1
del bola_data
for i in range(bba_data.shape[0]):
    for j in range(1, bba_data.shape[1]):
        past_action_index = int(bba_data[i, j, 12])
        past_th = bba_data[i, j, 4]
        past_buffer = bba_data[i, j-1, 10]
        all_past_ths = bba_data[i, j-1, 43:49]
        all_past_sizes = bba_data[i, j-1, 13:19]
        past_size = all_past_sizes[past_action_index]
        assert past_th == all_past_ths[past_action_index]
        mix_training_data[index, 0] = past_size
        mix_training_data[index, 1] = past_th
        mix_training_data[index, 2] = rtts[i]
        mix_training_data[index, 3] = past_buffer
        index = index + 1
del bba_data
size_mean, size_std = np.mean(mix_training_data[:, 0]), np.std(mix_training_data[:, 0])
th_mean, th_std = np.mean(mix_training_data[:, 1]), np.std(mix_training_data[:, 1])
rtt_mean, rtt_std = np.mean(rtts), np.std(rtts)
buff_mean, buff_std = np.mean(mix_training_data[:, 3]), np.std(mix_training_data[:, 3])
mix_locs = np.array([size_mean, th_mean+1.4*th_std, rtt_mean, buff_mean])
mix_scales = np.array([size_std, th_std, rtt_std, buff_std])
np.save(f'{mix_dir}/training_data.npy', mix_training_data)
np.save(f'{mix_dir}/locs.npy', mix_locs)
np.save(f'{mix_dir}/scales.npy', mix_scales)

base_mse = (bba_base_mse + bola_base_mse) / 2
np.save(f'{mix_dir}/cf_base_mse.npy', base_mse)