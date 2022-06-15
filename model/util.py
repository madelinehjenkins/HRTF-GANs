import numpy as np
import torch
from torch import autograd
import wandb
import os
import matplotlib.pyplot as plt
from torch import nn
import tifffile


# check for existing models and folders
def check_existence(tag):
    """Checks if model exists, then asks for user input. Returns True for overwrite, False for load.

    :param tag: [description]
    :type tag: [type]
    :raises SystemExit: [description]
    :raises AssertionError: [description]
    :return: True for overwrite, False for load
    :rtype: [type]
    """
    root = f'runs/{tag}'
    check_D = os.path.exists(f'{root}/Disc.pt')
    check_G = os.path.exists(f'{root}/Gen.pt')
    if check_G or check_D:
        print(f'Models already exist for tag {tag}.')
        x = input("To overwrite existing model enter 'o', to load existing model enter 'l' or to cancel enter 'c'.\n")
        if x == 'o':
            print("Overwriting")
            return True
        if x == 'l':
            print("Loading previous model")
            return False
        elif x == 'c':
            raise SystemExit
        else:
            raise AssertionError("Incorrect argument entered.")
    return True


# set-up util
def initialise_folders(tag, overwrite):
    """[summary]

    :param overwrite:
    :param tag: [description]
    :type tag: [type]
    """
    if overwrite:
        try:
            os.mkdir(f'runs')
        except:
            pass
        try:
            os.mkdir(f'runs/{tag}')
        except:
            pass


def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher, CUDAPrefetcher]:
    """Based on https://github.com/Lornatang/SRGAN-PyTorch/blob/main/train_srgan.py"""
    # Load train, test and valid datasets
    train_datasets = TrainValidImageDataset(config.train_image_dir, config.image_size, config.upscale_factor, "Train")
    valid_datasets = TrainValidImageDataset(config.valid_image_dir, config.image_size, config.upscale_factor, "Valid")
    test_datasets = TestImageDataset(config.test_lr_image_dir, config.test_hr_image_dir)

    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    valid_dataloader = DataLoader(valid_datasets,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=1,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, config.device)
    valid_prefetcher = CUDAPrefetcher(valid_dataloader, config.device)
    test_prefetcher = CUDAPrefetcher(test_dataloader, config.device)

    return train_prefetcher, valid_prefetcher, test_prefetcher


def calc_gradient_penalty(netD, real_data, fake_data, batch_size, l, device, gp_lambda, nc):
    """[summary]

    :param netD: [description]
    :type netD: [type]
    :param real_data: [description]
    :type real_data: [type]
    :param fake_data: [description]
    :type fake_data: [type]
    :param batch_size: [description]
    :type batch_size: [type]
    :param l: [description]
    :type l: [type]
    :param device: [description]
    :type device: [type]
    :param gp_lambda: [description]
    :type gp_lambda: [type]
    :param nc: [description]
    :type nc: [type]
    :return: [description]
    :rtype: [type]
    """
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(
        real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, nc, l, l)
    alpha = alpha.to(device)

    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(
                                  disc_interpolates.size()).to(device),
                              create_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty


def batch_real(img, l, bs):
    """[summary]
    :param training_imgs: [description]
    :type training_imgs: [type]
    :return: [description]
    :rtype: [type]
    """
    n_ph, x_max, y_max = img.shape
    data = torch.zeros((bs, n_ph, l, l))
    for i in range(bs):
        x, y = torch.randint(x_max - l, (1,)), torch.randint(y_max - l, (1,))
        data[i] = img[:, x:x + l, y:y + l]
    return data


# Evaluation util
def post_process(img):
    """Turns a n phase image (bs, n, imsize, imsize) into a plottable euler image (bs, 3, imsize, imsize, imsize)

    :param img: a tensor of the n phase img
    :type img: torch.Tensor
    :return:
    :rtype:
    """
    img = img.detach().cpu()
    img = torch.argmax(img, dim=1).unsqueeze(-1).numpy()

    return img * 255


def generate(c, netG):
    """Generate an instance from generator, save to .tif

    :param c: Config object class
    :type c: Config
    :param netG: Generator instance
    :type netG: Generator
    :return: Post-processed generated instance
    :rtype: torch.Tensor
    """
    tag, ngpu, nz, lf, pth = c.tag, c.ngpu, c.nz, c.lf, c.path

    out_pth = f"runs/{tag}/out.tif"
    if torch.cuda.device_count() > 1 and c.ngpu > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
    device = torch.device("cuda:0" if (
            torch.cuda.is_available() and ngpu > 0) else "cpu")
    if (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu))).to(device)
    netG.load_state_dict(torch.load(f"{pth}/Gen.pt"))
    netG.eval()
    noise = torch.randn(1, nz, lf, lf)
    raw = netG(noise)
    gb = post_process(raw)
    tif = np.array(gb[0], dtype=np.uint8)
    tifffile.imwrite(out_pth, tif, imagej=True)
    return tif


def progress(i, iters, n, num_epochs, timed):
    """[summary]

    :param i: [description]
    :type i: [type]
    :param iters: [description]
    :type iters: [type]
    :param n: [description]
    :type n: [type]
    :param num_epochs: [description]
    :type num_epochs: [type]
    :param timed: [description]
    :type timed: [type]
    """
    progress = 'iteration {} of {}, epoch {} of {}'.format(
        i, iters, n, num_epochs)
    print(f"Progress: {progress}, Time per iter: {timed}")


def plot_img(img, iter, epoch, path, offline=True):
    """[summary]

    :param img: [description]
    :type img: [type]
    :param slcs: [description], defaults to 4
    :type slcs: int, optional
    """
    img = post_process(img)
    if not offline:
        wandb.log({"slices": [wandb.Image(i) for i in img]})
    else:
        fig, axs = plt.subplots(1, img.shape[0])
        for ax, im in zip(axs, img):
            ax.imshow(im)
        plt.savefig(f'{path}/{epoch}_{iter}_slices.png')
