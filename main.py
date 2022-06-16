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
from preprocessing.utils import interpolate_fft_pad, generate_euclidean_cube, load_data
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

        # TODO: Split some data into test set
        train_sample = np.random.choice(ds.subject_ids, int(len(ds.subject_ids) * config.train_samples_ratio))
        for i in range(len(ds)):
            if i % 10 == 0:
                print(f"HRTF {i} out of {len(ds)} ({round(100 * i / len(ds))}%)")
            clean_hrtf = interpolate_fft_pad(cs, ds[i]['features'], sphere, sphere_triangles, sphere_coeffs, cube,
                                             config.hrtf_size, config.pad_width)
            # save cleaned hrtfdata
            if ds[i]['group'] in train_sample:
                data_dir = "projected_data/train/"
            else:
                data_dir = "projected_data/valid/"

            subject_id = str(ds[i]['group'])
            side = ds[i]['target']
            with open(data_dir + "ARI_" + subject_id + side, "wb") as file:
                pickle.dump(clean_hrtf, file)

    elif mode == 'train':
        # Trains the GANs, according to the parameters specified in Config

        train_prefetcher, valid_prefetcher = load_dataset(config)
        print("Loaded all datasets successfully.")

        overwrite = util.check_existence(tag)
        util.initialise_folders(tag, overwrite)
        train(config, train_prefetcher, overwrite=overwrite)


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

    # main('train', 'test', using_hpc=False)
