import numpy as np
import argparse
import pickle
parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="root directory")
parser.add_argument("--seed", type=int, help="random seed", default=0)
parser.add_argument("--num_samples", help="number of data points", default=int(1e7))
args = parser.parse_args()
np.random.seed(args.seed)

z = np.random.uniform(low=-0.5, high=0.5, size=(args.num_samples,))
epsilon_t = np.random.normal(loc=0.0, scale=1.0, size=(args.num_samples,))
epsilon_r = np.random.exponential(scale=1.0, size=(args.num_samples,))
epsilon_b = np.random.beta(a=1.0, b=1.0, size=(args.num_samples,))
weights = np.random.uniform(low=1.0, high=2.0, size=(3,))
biases = np.random.uniform(low=-1.0, high=1.0, size=(3,))
t = (weights[0] * z + biases[0] + epsilon_t) % (2 * np.pi)
r = 1 + np.multiply(np.exp(weights[1] * z + biases[1]), epsilon_r)
b = np.exp(weights[2] * z + biases[2]) + epsilon_b
y = np.multiply(b, 2 + np.sin(t))
x = np.multiply(np.multiply(r, b), 2 + np.cos(t))
dataset = {
    'data': {'z': z, 'epsilon_t': epsilon_t, 'epsilon_r': epsilon_r, 'epsilon_b': epsilon_b, 't': t, 'r': r, 'b': b,
             'weights': weights, 'biases': biases, 'x': x, 'y': y},
    'locs': {'z': np.mean(z), 't': np.mean(t), 'x': (np.min(x) + np.max(x)) / 2, 'y': (np.min(y) + np.max(y)) / 2},
    'scales': {'z': np.std(z), 't': np.std(t), 'x': (np.max(x) - np.min(x)) / 5.9, 'y': (np.max(y) - np.min(y)) / 5.9},
    'num_samples': args.num_samples
}
with open(f'{args.dir}dataset_{args.seed}.pkl', 'wb') as f:
    pickle.dump(dataset, f)
