import argparse
import os
import pickle
import torch
from hrtfdata.torch.full import ARI
import numpy as np

from config import Config
from model.train import train
from model.util import load_dataset
from preprocessing.cubed_sphere import CubedSphere
from preprocessing.utils import interpolate_fft, generate_euclidean_cube, load_data
from model import util

PI_4 = np.pi / 4

# Random seed to maintain reproducible results
torch.manual_seed(0)
np.random.seed(0)


def main(mode, tag, using_hpc):
    # Initialise Config object
    config = Config(tag, using_hpc=using_hpc)
    data_dir = config.raw_hrtf_dir / 'ARI'
    print(os.getcwd())
    print(data_dir)

    projection_filename = "projection_coordinates/ARI_projection_" + str(config.hrtf_size)
    if using_hpc:
        projection_filename = "HRTF-GANs/" + projection_filename

    if mode == 'generate_projection':
        # Must be run in this mode once per dataset, finds barycentric coordinates for each point in the cubed sphere

        # No need to load the entire dataset in this case
        ds: ARI = load_data(data_folder=data_dir, load_function=ARI, domain='time', side='left', subject_ids='first')
        # need to use protected member to get this data, no getters
        cs = CubedSphere(sphere_coords=ds._selected_angles)
        generate_euclidean_cube(cs.get_sphere_coords(), projection_filename, edge_len=config.hrtf_size)

    elif mode == 'preprocess':
        # Interpolates data to find HRIRs on cubed sphere, then FFT to obtain HRTF, finally splits data into train and
        # val sets and saves processed data

        ds: ARI = load_data(data_folder=data_dir, load_function=ARI, domain='time', side='both')
        # need to use protected member to get this data, no getters
        cs = CubedSphere(sphere_coords=ds._selected_angles)
        with open(projection_filename, "rb") as file:
            cube, sphere, sphere_triangles, sphere_coeffs = pickle.load(file)

        # Split data into train and test sets
        train_size = int(len(set(ds.subject_ids)) * config.train_samples_ratio)
        train_sample = np.random.choice(list(set(ds.subject_ids)), train_size, replace=False)

        # collect all train_hrtfs to get mean and sd
        train_hrtfs = torch.empty(size=(2 * train_size, 5, config.hrtf_size, config.hrtf_size, 128))
        j = 0
        for i in range(len(ds)):
            if i % 10 == 0:
                print(f"HRTF {i} out of {len(ds)} ({round(100 * i / len(ds))}%)")
            clean_hrtf = interpolate_fft(cs, ds[i]['features'], sphere, sphere_triangles, sphere_coeffs, cube,
                                         config.hrtf_size)
            # save cleaned hrtfdata
            if ds[i]['group'] in train_sample:
                projected_dir = "projected_data/train/"
                train_hrtfs[j] = clean_hrtf
                j += 1
            else:
                projected_dir = "projected_data/valid/"

            if using_hpc:
                projected_dir = "HRTF-GANs/" + projected_dir

            subject_id = str(ds[i]['group'])
            side = ds[i]['target']
            with open(projected_dir + "ARI_" + subject_id + side, "wb") as file:
                pickle.dump(clean_hrtf, file)

        # save dataset mean and standard deviation for each channel, across all HRTFs in the training data
        mean = torch.mean(train_hrtfs, [0, 1, 2, 3])
        std = torch.std(train_hrtfs, [0, 1, 2, 3])
        min_hrtf = torch.min(train_hrtfs)
        max_hrtf = torch.max(train_hrtfs)
        mean_std_filename = "projected_data/ARI_mean_std_min_max"
        if using_hpc:
            mean_std_filename = "HRTF-GANs/" + mean_std_filename
        with open(mean_std_filename, "wb") as file:
            pickle.dump((mean, std, min_hrtf, max_hrtf), file)

    elif mode == 'train':
        # with open(config.train_hrtf_dir + '/../ARI_mean_std_min_max', "rb") as file:
        #     mean, std, min_hrtf, max_hrtf = pickle.load(file)

        # Trains the GANs, according to the parameters specified in Config
        train_prefetcher, test_prefetcher = load_dataset(config, mean=None, std=None)
        print("Loaded all datasets successfully.")

        util.initialise_folders(tag, overwrite=True)
        train(config, train_prefetcher, overwrite=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("-t", "--tag")
    parser.add_argument("-c", "--hpc")
    args = parser.parse_args()

    if args.hpc == "True":
        hpc = True
    elif args.hpc == "False":
        hpc = False
    else:
        raise RuntimeError("Please enter 'True' or 'False' for the hpc tag (-c/--hpc)")

    if args.tag:
        tag = args.tag
    else:
        tag = 'test'
    main(args.mode, tag, hpc)

    # main('train', 'localtrain', using_hpc=False)
