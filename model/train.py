import scipy

from model.util import *
from model.model import *

import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import time

from plot import plot_losses, plot_magnitude_spectrums


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

    # get list of positive frequencies of HRTF for plotting magnitude spectrum
    hrir_samplerate = 48000.0
    all_freqs = scipy.fft.fftfreq(256, 1 / hrir_samplerate)
    pos_freqs = all_freqs[all_freqs >= 0]

    # Define Generator network and transfer to CUDA
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    if ('cuda' in str(device)) and (ngpu > 1):
        netD = (nn.DataParallel(netD, list(range(ngpu)))).to(device)
        netG = nn.DataParallel(netG, list(range(ngpu))).to(device)

    # Define optimizers
    optD = optim.Adam(netD.parameters(), lr=lr_dis, betas=(beta1, beta2))
    optG = optim.Adam(netG.parameters(), lr=lr_gen, betas=(beta1, beta2))

    # Define loss functions
    adversarial_criterion = nn.BCEWithLogitsLoss()
    content_criterion = spectral_distortion_metric

    if not overwrite:
        netG.load_state_dict(torch.load(f"{path}/Gen.pt"))
        netD.load_state_dict(torch.load(f"{path}/Disc.pt"))

    train_losses_G = []
    train_losses_G_adversarial = []
    train_losses_G_content = []
    train_losses_D = []
    train_losses_D_hr = []
    train_losses_D_sr = []

    for epoch in range(num_epochs):
        times = []
        train_loss_G = 0.
        train_loss_G_adversarial = 0.
        train_loss_G_content = 0.
        train_loss_D = 0.
        train_loss_D_hr = 0.
        train_loss_D_sr = 0.

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
            lr = batch_data["lr"].to(device=device, memory_format=torch.contiguous_format,
                                     non_blocking=True, dtype=torch.float)
            hr = batch_data["hr"].to(device=device, memory_format=torch.contiguous_format,
                                     non_blocking=True, dtype=torch.float)

            # during every 25th epoch and last epoch, save filename for mag spectrum plot
            if epoch % 25 == 0 or epoch == (num_epochs - 1):
                filename = batch_data["filename"]

            # Discriminator Training
            # Initialize the discriminator model gradients
            netD.zero_grad()

            # Use the generator model to generate fake samples
            sr = netG(lr)

            # Calculate the classification score of the discriminator model for real samples
            label = torch.full((batch_size, ), 1., dtype=hr.dtype, device=device)
            output = netD(hr).view(-1)
            loss_D_hr = adversarial_criterion(output, label)
            loss_D_hr.backward()

            # train on SR hrtfs
            label.fill_(0.)
            output = netD(sr.detach().clone()).view(-1)
            loss_D_sr = adversarial_criterion(output, label)
            loss_D_sr.backward()

            # Compute the discriminator loss
            loss_D = loss_D_hr + loss_D_sr
            train_loss_D += loss_D.item()
            train_loss_D_hr += loss_D_hr.item()
            train_loss_D_sr += loss_D_sr.item()

            # Update D
            optD.step()

            # Generator training
            if batch_index % int(critic_iters) == 0:
                # Initialize generator model gradients
                netG.zero_grad()
                sr = netG(lr)
                label.fill_(1.)
                # Calculate adversarial loss
                output = netD(sr).view(-1)

                content_loss_G = config.content_weight * content_criterion(sr, hr)
                adversarial_loss_G = config.adversarial_weight * adversarial_criterion(output, label)

                print(f'Output shape: {sr.shape}')
                print(f'Output[0] shape: {sr[0].shape}')
                print(f'Label shape: {hr.shape}')
                print(f'Loss shape: {content_loss_G.shape}')
                print(f'Content Loss: {content_loss_G}')

                # Calculate the generator total loss value and backprop
                loss_G = content_loss_G + adversarial_loss_G
                loss_G.backward()
                print(f'Loss item: {loss_G.item()}')

                train_loss_G += loss_G.item()
                train_loss_G_adversarial += adversarial_loss_G.item()
                train_loss_G_content += content_loss_G.item()

                optG.step()

            if ('cuda' in str(device)) and (ngpu > 1):
                end_overall.record()
                torch.cuda.synchronize()
                times.append(start_overall.elapsed_time(end_overall))
            else:
                end_overall = time.time()
                times.append(end_overall - start_overall)

            # Every 0th batch log useful metrics
            if batch_index == 0:
                with torch.no_grad():
                    torch.save(netG.state_dict(), f'{path}/Gen.pt')
                    torch.save(netD.state_dict(), f'{path}/Disc.pt')

                    progress(batch_index, batches, epoch, num_epochs, timed=np.mean(times))
                    times = []

            # Preload the next batch of data
            batch_data = train_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the
            # terminal print data normally
            batch_index += 1

        train_losses_D.append(train_loss_D / len(train_prefetcher))
        train_losses_D_hr.append(train_loss_D_hr / len(train_prefetcher))
        train_losses_D_sr.append(train_loss_D_sr / len(train_prefetcher))
        train_losses_G.append(train_loss_G / len(train_prefetcher))
        train_losses_G_adversarial.append(train_loss_G_adversarial / len(train_prefetcher))
        train_losses_G_content.append(train_loss_G_content / len(train_prefetcher))
        print(f"Average epoch loss, discriminator: {train_losses_D[-1]}, generator: {train_losses_G[-1]}")
        print(f"Average epoch loss, D_real: {train_losses_D_hr[-1]}, D_fake: {train_losses_D_sr[-1]}")
        print(f"Average epoch loss, G_adv: {train_losses_G_adversarial[-1]}, train_losses_G_content: {train_losses_G_content[-1]}")

        # create magnitude spectrum plot every 25 epochs and last epoch
        if epoch % 25 == 0 or epoch == (num_epochs - 1):
            i_plot = 0
            magnitudes_real = torch.permute(hr.detach().cpu()[i_plot], (1, 2, 3, 0))
            magnitudes_interpolated = torch.permute(sr.detach().cpu()[i_plot], (1, 2, 3, 0))

            if filename[i_plot][-5:] == 'right':
                ear_label = 'right'
            elif filename[i_plot][-4:] == 'left':
                ear_label = 'left'
            else:
                ear_label = 'unknown'

            plot_label = filename[i_plot].split('/')[-1] + '_epoch' + epoch
            plot_magnitude_spectrums(pos_freqs, magnitudes_real, magnitudes_interpolated,
                                     ear_label, "training", plot_label, path, log_scale_magnitudes=True)

    plot_losses(train_losses_D, train_losses_G,
                label_1='Discriminator loss', label_2='Generator loss',
                path=path, filename='loss_curves')
    plot_losses(train_losses_D_hr, train_losses_D_sr,
                label_1='Discriminator loss, real', label_2='Discriminator loss, fake',
                path=path, filename='loss_curves_D')
    plot_losses(train_losses_G_adversarial, train_losses_G_content,
                label_1='Generator loss, adversarial', label_2='Generator loss, content',
                path=path, filename='loss_curves_G')

    print("TRAINING FINISHED")
