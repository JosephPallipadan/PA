from os import listdir

import torch
import torch.utils.data

from torch import nn, optim
from torch.nn import functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import sqrt

from MVAE import MVAE
from amc_parser import parse_amc

from pprint import pprint
from logger import logger
import matplotlib.pyplot as plt

torch.manual_seed(0)
batch_size = 64
log_interval = 10
epochs = 24
device = torch.device("cuda")


class MotionFramesDataset(Dataset):
    def __init__(self, subject_count, max_files_per_subject):
        all_subject_motions = self.get_motion_lists(subject_count, max_files_per_subject)
        self.samples = []

        for subject_motion_list in all_subject_motions:
            for frame_list in subject_motion_list:
                for i in range(1, len(frame_list)):

                    previous_frame, current_frame = frame_list[i - 1], frame_list[i]
                    current_frame_feature_list, next_frame_feature_list = [], []

                    for current_frame_feature, next_frame_feature in zip(previous_frame.values(), current_frame.values()):
                        current_frame_feature_list += current_frame_feature
                        next_frame_feature_list += next_frame_feature

                    self.samples.append((torch.Tensor(current_frame_feature_list), torch.Tensor(next_frame_feature_list)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_motion_lists(self, subject_count, max_files):
        all_subject_motions = []
        subject_number = 35

        while subject_count > 0:

            current_subject_frame_list = []
            files = listdir(f'data/{subject_number}')

            for file_name in files[1:]:

                frame = parse_amc(f'data/{subject_number}/{file_name}')
                current_subject_frame_list.append(frame)

                max_files -= 1
                if max_files == 0:
                    break
                
            subject_count -= 1
            all_subject_motions.append(current_subject_frame_list)
        return all_subject_motions


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # BCE_function = nn.BCEWithLogitsLoss()
    # BCE = BCE_function(recon_x, x)
    recon_x = F.sigmoid(recon_x)
    recon_loss = F.mse_loss(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + 0.2*KLD


def train(epoch, model, train_loader, optimizer):
    losses = []
    model.train()
    train_loss = 0
    for (index, data) in enumerate(train_loader):
        previous_frame, current_frame = torch.flatten(data[0]).to(device), torch.flatten(data[1]).to(device)

        previous_frame = (previous_frame - previous_frame.mean()) / sqrt(previous_frame.var())
        current_frame = (current_frame - current_frame.mean()) / sqrt(current_frame.var())

        optimizer.zero_grad()
        reconstructed_next_frame, mu, logvar = model(previous_frame, current_frame)

        loss = loss_function(reconstructed_next_frame, current_frame, mu, logvar)
        loss.backward()

        train_loss += loss.item()
        losses.append(loss.item())
        optimizer.step()
        logger.info(f'Iteration {index+1} of {len(train_loader.dataset)} | Loss: {loss.item()}')
    
    plt.plot(losses)
    plt.ylabel('Loss')
    plt.xlabel('Iteration No.')
    plt.savefig("Fig.png", dpi=500)

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    logger.info('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    logger.info("Initializing Dataset...")
    dataset = MotionFramesDataset(subject_count=1, max_files_per_subject=34)
    logger.info("Done.\n")

    logger.info("Loading Training Data...")
    train_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=True)
    logger.info("Done.\n")

    logger.info("Initializing Network and Optimizer...")
    model = MVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    logger.info("Done.\n")


    train(0, model, train_loader, optimizer)
   