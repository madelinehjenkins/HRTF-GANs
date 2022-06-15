import argparse
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


def load_data(data_folder, load_function, domain, side, subject_ids=None):
    base_dir = Path('/Users/madsjenkins/Imperial/HRTF/Volumes/home/HRTF Datasets')
    if subject_ids:
        return load_function(base_dir / data_folder,
                             feature_spec={"hrirs": {'side': side, 'domain': domain}},
                             target_spec={"side": {}},
                             group_spec={"subject": {}}, subject_ids="last")
    return load_function(base_dir / data_folder,
                         feature_spec={"hrirs": {'side': side, 'domain': domain}},
                         target_spec={"side": {}},
                         group_spec={"subject": {}})


def main(mode, tag):
    if mode == 'generate_projection':
        ds: ARI = load_data(data_folder='ARI', load_function=ARI, domain='time', side='left', subject_ids='first')
        # need to use protected member to get this data, no getters
        cs = CubedSphere(sphere_coords=ds._selected_angles)
        generate_euclidean_cube(cs.get_sphere_coords(), "projection_coordinates/ARI_projection_16", edge_len=16)

    elif mode == 'preprocess':
        ds: ARI = load_data(data_folder='ARI', load_function=ARI, domain='time', side='both')
        # need to use protected member to get this data, no getters
        cs = CubedSphere(sphere_coords=ds._selected_angles)
        with open("projection_coordinates/ARI_projection_16", "rb") as file:
            cube, sphere, sphere_triangles, sphere_coeffs = pickle.load(file)

        edge_len = 16
        pad_width = 2
        train_samples_ratio = 0.8
        train_sample = np.random.choice(ds.subject_ids, int(len(ds.subject_ids) * train_samples_ratio))
        for i in range(len(ds)):
            if i % 10 == 0:
                print(f"HRTF {i} out of {len(ds)} ({round(100 * i / len(ds))}%)")
            clean_hrtf = interpolate_fft_pad(cs, ds[i]['features'], sphere, sphere_triangles, sphere_coeffs, cube,
                                             edge_len, pad_width)
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
        # Initialise Config object
        config = Config(tag)
        train_prefetcher, valid_prefetcher = load_dataset(config)
        print("Loaded all datasets successfully.")

        overwrite = util.check_existence(tag)
        util.initialise_folders(tag, overwrite)
        train(config, train_prefetcher, overwrite=overwrite)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("mode")
    # parser.add_argument("-t", "--tag")
    # args = parser.parse_args()
    # if args.tag:
    #     tag = args.tag
    # else:
    #     tag = 'test'
    # main(args.mode, tag)
    main('train', 'test')
