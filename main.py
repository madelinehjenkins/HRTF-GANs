import itertools

import torch
from hrtfdata.torch.full import ARI
from hrtfdata.torch import collate_dict_dataset
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


def load_data(data_folder, load_function, domain, side):
    base_dir = Path('/Users/madsjenkins/Imperial/HRTF/Volumes/home/HRTF Datasets')
    return load_function(base_dir / data_folder,
                         feature_spec={"hrirs": {'side': side, 'domain': domain}},
                         target_spec={"side": {}},
                         group_spec={"subject": {}},
                         subject_ids="first")  # temporary measure to avoid loading entire dataset each time


def get_panel(latitude, longitude):
    if latitude is None or longitude is None:
        return None
    # when close to the horizontal plane, must be panels 1 through 4 (inclusive)
    if -np.pi / 4 <= latitude <= np.pi / 4:
        if -np.pi / 4 <= longitude <= np.pi / 4:
            return 1
        elif np.pi / 4 <= longitude <= 3 * np.pi / 4:
            return 2
        elif 3 * np.pi / 4 <= longitude or longitude <= -3 * np.pi / 4:
            return 3
        elif -3 * np.pi / 4 <= longitude <= -np.pi / 4:
            return 4
    # above a certain latitude, in panel 5
    elif latitude > np.pi / 4:
        return 5
    # below a certain latitude, in panel 6
    elif latitude < -np.pi / 4:
        return 6


def get_cube_coords(latitude, longitude):
    panel = get_panel(latitude, longitude)
    if panel is None:
        # TODO: come up with a more sensible way of handling these
        panel, x, y = np.nan, np.nan, np.nan
    else:
        if panel <= 4:
            offset = (((panel - 1) / 2) * np.pi)
            x = longitude - offset
            y = np.arctan(np.tan(latitude) / np.cos(longitude - offset))
        elif panel == 5:
            x = np.arctan(np.sin(longitude) / np.tan(latitude))
            y = np.arctan(-np.cos(longitude) / np.tan(latitude))
        elif panel == 6:
            x = np.arctan(-np.sin(longitude) / np.tan(latitude))
            y = np.arctan(-np.cos(longitude) / np.tan(latitude))
    return panel, x, y


def plot_sphere(sphere_coords):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Format data.
    x, y, z = [], [], []

    for latitude, longitude in sphere_coords:
        if latitude is not None and longitude is not None:
            # convert to cartesian coordinates
            x_i = np.cos(latitude) * np.cos(longitude)
            y_i = np.cos(latitude) * np.sin(longitude)
            z_i = np.sin(latitude)

            x.append(x_i)
            y.append(y_i)
            z.append(z_i)

    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)

    # Plot the surface.
    surf = ax.scatter(x, y, z, cmap=cm.coolwarm,
                      linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    plt.show()


class CubedSphere(object):
    def __init__(self, sphere_coords):
        self.sphere_coords = []
        for proj_angle_i in sphere_coords.keys():
            # convert degrees to radians by multiplying by a factor of pi/180
            vert_angles_i = sphere_coords[proj_angle_i] * np.pi / 180
            proj_angle_i = proj_angle_i * np.pi / 180

            num_measurements = vert_angles_i.shape[0]
            # measurement_positions is stored as (latitude, longitude)
            self.sphere_coords += list(zip(vert_angles_i.tolist(), [proj_angle_i] * num_measurements))

        # self.cube_coords is created from measurement_positions, such that order is the same
        self.cube_coords = list(itertools.starmap(get_cube_coords, self.sphere_coords))
        print(f"measurement_positions: {self.sphere_coords[1500]}")
        print(f"measurement_positions len: {len(self.sphere_coords)}")
        print(f"cube_coords: {self.cube_coords[1500]}")
        print(f"cube_coords shape: {len(self.cube_coords)}")

    def get_sphere_coords(self):
        return self.sphere_coords


def main():
    ds: ARI = load_data(data_folder='ARI', load_function=ARI, domain='magnitude_db', side='left')

    # need to use protected member to get this data, no getters
    cs = CubedSphere(sphere_coords=ds._selected_angles)
    plot_sphere(cs.get_sphere_coords())


if __name__ == '__main__':
    main()
