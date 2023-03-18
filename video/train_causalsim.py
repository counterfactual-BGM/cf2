import os
import numpy as np
import argparse
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="root directory")
parser.add_argument("--seed", type=int, help="seed")
parser.add_argument("--batch_size", type=int, help="log batch size")
parser.add_argument("--device", type=str, help="Compute device", default='cuda:0')
parser.add_argument("--C", type=float, help="discriminator loss coefficient")
parser.add_argument("--start_epoch", type=int, default=0, help="epoch number to start with")
parser.add_argument('--no_val', action='store_true', help='skip validation')

args = parser.parse_args()
DEVICE = torch.device(args.device)
NUM_EPOCHS = 30
BATCH_SIZE = 2 ** args.batch_size
model_dir = f'{args.dir}iv/models'
os.makedirs(model_dir, exist_ok=True)
DISC_IT = 10
mse_loss = nn.MSELoss()
ce_loss = nn.CrossEntropyLoss()


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_sizes, activation):
        super().__init__()
        self.predict = mlp(sizes=[input_dim] + list(hidden_sizes) + [output_dim], activation=activation,
                           output_activation=nn.Identity)

    def forward(self, raw_input):
        prediction = self.predict(raw_input)
        return prediction


training_data = np.load(f'{args.dir}iv/training_data.npy')
locs = np.load(f'{args.dir}iv/locs.npy')
scales = np.load(f'{args.dir}iv/scales.npy')
cf_data = np.load(f'{args.dir}iv/cf_data.npy')
base_mse = np.load(f'{args.dir}iv/cf_base_mse.npy')

data_size = training_data.shape[0]
start_indices = np.arange(0, data_size, BATCH_SIZE)
end_indices = np.append(start_indices[1:], [data_size])
if args.start_epoch > 0:
    encoder = torch.load(f'{model_dir}/C{args.C}_seed{args.seed}_encoder_{args.start_epoch}.pth')
    decoder = torch.load(f'{model_dir}/C{args.C}_seed{args.seed}_decoder_{args.start_epoch}.pth')
    disc = torch.load(f'{model_dir}/C{args.C}_seed{args.seed}_disc_{args.start_epoch}.pth')
else:
    encoder = MLP(input_dim=3, output_dim=1, hidden_sizes=[128, 128], activation=nn.ReLU).to(device=DEVICE)
    decoder = MLP(input_dim=3, output_dim=1, hidden_sizes=[128, 128], activation=nn.ReLU).to(device=DEVICE)
    disc = MLP(input_dim=11, output_dim=2, hidden_sizes=[128, 128], activation=nn.ReLU).to(device=DEVICE)
encoder_optimizer = torch.optim.Adam(encoder.parameters())
decoder_optimizer = torch.optim.Adam(decoder.parameters())
disc_optimizer = torch.optim.Adam(disc.parameters())
writer = SummaryWriter(log_dir=f"{args.dir}iv/logs/C{args.C}_seed{args.seed}")
actions = torch.as_tensor((training_data[:, 0:1] - locs[0]) / scales[0], dtype=torch.float32, device=DEVICE)
rtts = torch.as_tensor((training_data[:, 2:3] - locs[2]) / scales[2], dtype=torch.float32, device=DEVICE)
algo_indices = torch.as_tensor(training_data[:, 3], dtype=torch.long, device=DEVICE)
algo_one_hot = nn.functional.one_hot(algo_indices)
obs = torch.as_tensor((training_data[:, 1:2] - locs[1]) / scales[1], dtype=torch.float32, device=DEVICE)
del training_data, algo_indices

cf_actions = torch.as_tensor((cf_data[:, 0:6] - locs[0]) / scales[0], dtype=torch.float32, device=DEVICE)
cf_ths = torch.as_tensor(cf_data[:, 6:12], dtype=torch.float32, device=DEVICE)

cf_actions = torch.cat([cf_actions for _ in range(9)], dim=0)
cf_ths = torch.cat([cf_ths for _ in range(9)], dim=0)
del cf_data
for i in tqdm(range(args.start_epoch, NUM_EPOCHS)):
    disc_loss_array = []
    rec_loss_array = []
    all_loss_array = []
    loss_weights = []
    rnd_indices = torch.randperm(data_size, dtype=torch.long, device=DEVICE)
    permuted_obs = obs[rnd_indices]
    permuted_actions = actions[rnd_indices]
    permuted_rtts = rtts[rnd_indices]
    permuted_algos = algo_one_hot[rnd_indices]
    for start_idx, end_idx in zip(start_indices, end_indices):
        # ########### Start Discriminator Training ###########
        batch_rtts = permuted_rtts[start_idx:end_idx]
        batch_algos = permuted_algos[start_idx:end_idx]
        batch_size = end_idx - start_idx
        loss_weights.append(batch_size)
        batch_labels = torch.cat([torch.ones(size=(batch_size, 1), device=DEVICE, dtype=torch.long),
                                  torch.zeros(size=(batch_size, 1), device=DEVICE, dtype=torch.long)], dim=0)[:, 0]
        with torch.no_grad():
            features = encoder(torch.cat(
                [batch_rtts, permuted_actions[start_idx:end_idx], permuted_obs[start_idx:end_idx]], dim=1))
        for disc_it in range(DISC_IT):
            disc_optimizer.zero_grad()
            joint_disc_input = torch.cat([features, batch_rtts, batch_algos], dim=1)
            batch_rnd_indices = torch.randperm(batch_size, dtype=torch.long, device=DEVICE)
            permuted_batch_algos = batch_algos[batch_rnd_indices]
            permuted_batch_rtts = batch_rtts[batch_rnd_indices]
            marginal_disc_input = torch.cat([features, permuted_batch_rtts, permuted_batch_algos], dim=1)
            all_disc_input = torch.cat([joint_disc_input, marginal_disc_input], dim=0)
            discriminated_tensors = disc(all_disc_input)
            assert batch_labels.shape[0] == discriminated_tensors.shape[0]
            disc_loss = ce_loss(discriminated_tensors, batch_labels)
            disc_loss.backward()
            disc_optimizer.step()
        # ########### Start Reconstruction Training ###########
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        features = encoder(torch.cat(
            [batch_rtts, permuted_actions[start_idx:end_idx], permuted_obs[start_idx:end_idx]], dim=1))
        pred_obs = decoder(torch.cat([batch_rtts, permuted_actions[start_idx:end_idx], features], dim=1))
        assert pred_obs.shape == permuted_obs[start_idx:end_idx].shape
        pred_loss = mse_loss(pred_obs, permuted_obs[start_idx:end_idx])
        joint_disc_input = torch.cat([features, batch_rtts, batch_algos], dim=1)
        batch_rnd_indices = torch.randperm(batch_size, dtype=torch.long, device=DEVICE)
        permuted_batch_algos = batch_algos[batch_rnd_indices]
        permuted_batch_rtts = batch_rtts[batch_rnd_indices]
        marginal_disc_input = torch.cat([features, permuted_batch_rtts, permuted_batch_algos], dim=1)
        all_disc_input = torch.cat([joint_disc_input, marginal_disc_input], dim=0)
        discriminated_tensors = disc(all_disc_input)
        assert batch_labels.shape[0] == discriminated_tensors.shape[0]
        disc_loss = ce_loss(discriminated_tensors, batch_labels)
        total_loss = pred_loss - args.C * disc_loss
        total_loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        disc_loss_array.append(disc_loss.detach().cpu().item())
        rec_loss_array.append(pred_loss.detach().cpu().item())
        all_loss_array.append(total_loss.detach().cpu().item())
    writer.add_scalar('adv/discrimination_loss', np.average(disc_loss_array, weights=loss_weights), i)
    writer.add_scalar('adv/reconstruction_loss', np.average(rec_loss_array, weights=loss_weights), i)
    writer.add_scalar('adv/total_loss', np.average(all_loss_array, weights=loss_weights), i)
    if args.no_val:
        pass
    else:
        if i % 10 == 0:
            mse_list = []
            mse_weights = []
            for start_idx, end_idx in zip(start_indices, end_indices):
                with torch.no_grad():
                    features = encoder(torch.cat(
                        [rtts[start_idx:end_idx], actions[start_idx:end_idx], obs[start_idx:end_idx]], dim=1))
                    for j in range(6):
                        cf_context_tensors = torch.cat([cf_actions[start_idx:end_idx, j:j + 1], rtts[start_idx:end_idx]],
                                                       dim=1)
                        pred_obs = decoder(
                            torch.cat([rtts[start_idx:end_idx], cf_actions[start_idx:end_idx, j:j + 1], features], dim=1))
                        pred_obs = pred_obs * scales[1] + locs[1]
                        assert pred_obs.shape == cf_ths[start_idx:end_idx, j:j + 1].shape
                        mse_list.append(
                            torch.mean(torch.square(pred_obs - cf_ths[start_idx:end_idx, j:j + 1])).cpu().item())
                        mse_weights.append(end_idx - start_idx)
            final_mse = np.average(mse_list, weights=mse_weights)
            r2 = 1 - (final_mse / base_mse)
            writer.add_scalar('r2', r2, i)
torch.save(encoder, f'{model_dir}/C{args.C}_seed{args.seed}_encoder_{args.start_epoch}.pth')
torch.save(decoder, f'{model_dir}/C{args.C}_seed{args.seed}_decoder_{args.start_epoch}.pth')
torch.save(disc, f'{model_dir}/C{args.C}_seed{args.seed}_disc_{args.start_epoch}.pth')
