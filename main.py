import argparse
import pickle
import torch
from hrtfdata.torch.full import ARI
from pathlib import Path
import numpy as np

from model.config import Config
from model.train import train
from preprocessing.cubed_sphere import CubedSphere
from plot import plot_padded_panels
from preprocessing.utils import interpolate_fft_pad, generate_euclidean_cube
from model import util, model

PI_4 = np.pi / 4


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
            load_cube, load_sphere, load_sphere_triangles, load_sphere_coeffs = pickle.load(file)

        edge_len = 16
        pad_width = 2
        all_subects = []
        for subject in range(len(ds)):
            if subject % 10 == 0:
                print(f"Subject {subject} out of {len(ds)} ({round(100 * subject / len(ds))}%)")
            all_subects.append(interpolate_fft_pad(cs, ds[subject],
                                                   load_sphere, load_sphere_triangles, load_sphere_coeffs, load_cube,
                                                   edge_len, pad_width))

        # save all_subjects
        with open("projected_data/ARI_processed_data", "wb") as file:
            pickle.dump(all_subects, file)

    elif mode == 'train':
        with open("projected_data/ARI_processed_data", "rb") as file:
            all_subects = pickle.load(file)


        # Initialise Config object
        config = Config(tag)
        overwrite = util.check_existence(tag)
        util.initialise_folders(tag, overwrite)
        train(config, prefetcher, overwrite=overwrite)


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
