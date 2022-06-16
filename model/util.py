import torch
import os

# check for existing models and folders
from torch.utils.data import DataLoader
from model.dataset import CUDAPrefetcher, TrainValidHRTFDataset, CPUPrefetcher


def check_existence(tag):
    """Checks if model exists, then asks for user input. Returns True for overwrite, False for load.

    :param tag: label to use for run
    :raises SystemExit: Raises if user cancels overwrite
    :raises AssertionError: Raises if user provides invalid input
    :return: True for overwrite, False for load
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


def initialise_folders(tag, overwrite):
    """Set up folders for given tag

    :param tag: label to use for run
    :param overwrite: whether to overwrite existing model outputs
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


def load_dataset(config) -> [CUDAPrefetcher, CUDAPrefetcher, CUDAPrefetcher]:
    """Based on https://github.com/Lornatang/SRGAN-PyTorch/blob/main/train_srgan.py"""
    # Load train, test and valid datasets
    train_datasets = TrainValidHRTFDataset(config.train_hrtf_dir, config.hrtf_size, config.upscale_factor, "Train")
    valid_datasets = TrainValidHRTFDataset(config.valid_hrtf_dir, config.hrtf_size, config.upscale_factor, "Valid")
    # TODO: set up test datasets
    # test_datasets = TestHRTFDataset(config.test_lr_hrtf_dir, config.test_hr_hrtf_dir)

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
    # test_dataloader = DataLoader(test_datasets,
    #                              batch_size=1,
    #                              shuffle=False,
    #                              num_workers=1,
    #                              pin_memory=True,
    #                              drop_last=False,
    #                              persistent_workers=True)

    # Place all data on the preprocessing data loader
    if torch.cuda.is_available() and config.ngpu > 0:
        device = torch.device(config.device_name)
        train_prefetcher = CUDAPrefetcher(train_dataloader, device)
        valid_prefetcher = CUDAPrefetcher(valid_dataloader, device)
        # test_prefetcher = CUDAPrefetcher(test_dataloader, device)
    else:
        train_prefetcher = CPUPrefetcher(train_dataloader)
        valid_prefetcher = CPUPrefetcher(valid_dataloader)

    return train_prefetcher, valid_prefetcher  # , test_prefetcher


def progress(i, batches, n, num_epochs, timed):
    """Prints progress to console

    :param i: Batch index
    :param batches: total number of batches
    :param n: Epoch number
    :param num_epochs: Total number of epochs
    :param timed: Time per batch
    """
    message = 'batch {} of {}, epoch {} of {}'.format(i, batches, n, num_epochs)
    print(f"Progress: {message}, Time per iter: {timed}")
