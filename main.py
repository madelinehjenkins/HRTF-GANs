import pickle
import torch
from hrtfdata.torch.full import ARI, CHEDAR
from pathlib import Path
import numpy as np

from cubed_sphere import CubedSphere
from plot import plot_padded_panels
from utils import interpolate_fft_pad

PI_4 = np.pi / 4


def load_data(data_folder, load_function, domain, side):
    base_dir = Path('/Users/madsjenkins/Imperial/HRTF/Volumes/home/HRTF Datasets')
    return load_function(base_dir / data_folder,
                         feature_spec={"hrirs": {'side': side, 'domain': domain}},
                         target_spec={"side": {}},
                         group_spec={"subject": {}}, subject_ids="last")


def main():
    ds: ARI = load_data(data_folder='ARI', load_function=ARI, domain='time', side='both')
    # need to use protected member to get this data, no getters
    cs = CubedSphere(sphere_coords=ds._selected_angles)

    # generate_euclidean_cube(cs.get_sphere_coords(), "ARI_projection_16", edge_len=16)

    with open("ARI_projection_16", "rb") as file:
        load_cube, load_sphere, load_sphere_triangles, load_sphere_coeffs = pickle.load(file)

    edge_len = 16
    pad_width = 2
    all_subects = []
    for subject in range(len(ds)):
        print(f"Subject {subject} out of {len(ds)} ({round(100 * subject / len(ds))}%)")
        all_subects.append(interpolate_fft_pad(cs, ds[subject],
                                               load_sphere, load_sphere_triangles, load_sphere_coeffs, load_cube,
                                               edge_len, pad_width))

    for mag_tensor in all_subects:
        plot_padded_panels(torch.select(mag_tensor, 3, 5), edge_len, pad_width=pad_width,
                           label_cells=False,
                           title=f"All cube faces, with padded areas shown outside hashes")


if __name__ == '__main__':
    main()
