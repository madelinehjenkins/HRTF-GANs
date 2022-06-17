from model.util import *
from model.model import *

import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time


def train(config, train_prefetcher, overwrite=True):
    """ Train the generator and discriminator models

    :param config: Config object containing model hyperparameters
    :param train_prefetcher: prefetcher for training data
    :param overwrite: whether to overwrite existing model outputs
    """
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)

    # Assign torch device
    ngpu = config.ngpu
    path = config.path
    device = torch.device(config.device_name if (
            torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(f'Using {ngpu} GPUs')
    print(device, " will be used.\n")
    cudnn.benchmark = True

    # Get train params
    batch_size, beta1, beta2, num_epochs, lr_gen, lr_dis, critic_iters = config.get_train_params()

    # Define Generator network and transfer to CUDA
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    if ('cuda' in str(device)) and (ngpu > 1):
        netD = (nn.DataParallel(netD, list(range(ngpu)))).to(device)
        netG = nn.DataParallel(netG, list(range(ngpu))).to(device)
    # Define optimizers
    optD = optim.Adam(netD.parameters(), lr=lr_dis, betas=(beta1, beta2))
    optG = optim.Adam(netG.parameters(), lr=lr_gen, betas=(beta1, beta2))

    if not overwrite:
        netG.load_state_dict(torch.load(f"{path}/Gen.pt"))
        netD.load_state_dict(torch.load(f"{path}/Disc.pt"))

    train_losses_G = []
    train_losses_D = []

    for epoch in range(num_epochs):
        times = []
        train_loss_D = 0.
        train_loss_G = 0.

        # Initialize the number of data batches to print logs on the terminal
        batch_index = 0

        # Initialize the data loader and load the first batch of data
        train_prefetcher.reset()
        batch_data = train_prefetcher.next()

        while batch_data is not None:
            if ('cuda' in str(device)) and (ngpu > 1):
                start_overall = torch.cuda.Event(enable_timing=True)
                end_overall = torch.cuda.Event(enable_timing=True)
                start_overall.record()
            else:
                start_overall = time.time()

            # Transfer in-memory data to CUDA devices to speed up training
            lr = batch_data["lr"].to(device=device, memory_format=torch.contiguous_format, non_blocking=True)
            hr = batch_data["hr"].to(device=device, memory_format=torch.contiguous_format, non_blocking=True)

            # Discriminator Training
            # Initialize the discriminator model gradients
            netD.zero_grad()

            # Use the generator model to generate fake samples
            sr = netG(lr)

            # Calculate the classification score of the discriminator model for real samples
            hr_output = netD(hr).mean()

            # train on SR hrtfs
            sr_output = netD(sr.detach()).mean()

            # Compute the discriminator loss and backprop
            disc_cost = sr_output - hr_output
            train_loss_D += disc_cost
            disc_cost.backward()

            optD.step()

            # Generator training
            if batch_index % int(critic_iters) == 0:
                # Initialize generator model gradients
                netG.zero_grad()
                # Calculate adversarial loss
                output = -netD(sr).mean()
                train_loss_G += output

                # Calculate loss for G and backprop
                output.backward()
                optG.step()

            if ('cuda' in str(device)) and (ngpu > 1):
                end_overall.record()
                torch.cuda.synchronize()
                times.append(start_overall.elapsed_time(end_overall))
            else:
                end_overall = time.time()
                times.append(end_overall - start_overall)

            # Every 5 batches log useful metrics
            if batch_index % 5 == 0:
                with torch.no_grad():
                    torch.save(netG.state_dict(), f'{path}/Gen.pt')
                    torch.save(netD.state_dict(), f'{path}/Disc.pt')
                    progress(batch_index, batches, epoch, num_epochs,
                             timed=np.mean(times))
                    times = []

            # Preload the next batch of data
            batch_data = train_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the
            # terminal print data normally
            batch_index += 1

        train_losses_D.append(train_loss_D / len(train_prefetcher))
        train_losses_G.append(train_loss_G / len(train_prefetcher))
        print(f"Average epoch loss, discriminator: {train_losses_D[-1]}, generator: {train_losses_G[-1]}")

    print("TRAINING FINISHED")
