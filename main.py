from hrtfdata.torch.full import ARI
from hrtfdata.torch import collate_dict_dataset
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np


def load_data(data_folder, load_function, domain, side):
    base_dir = Path('/Users/madsjenkins/Imperial/HRTF/Volumes/home/HRTF Datasets')
    return load_function(base_dir / data_folder,
                         feature_spec={"hrirs": {'side': side, 'domain': domain}},
                         target_spec={"side": {}},
                         group_spec={"subject": {}},
                         subject_ids="first")  # temporary measure to avoid loading entire dataset each time


def get_panel(latitude, longitude):
    # when close to the horizontal plane, must be panels 1 through 4 (inclusive)
    if -np.pi / 4 <= latitude <= np.pi/4:
        if -np.pi / 4 <= longitude <= np.pi/4:
            return 1
        elif np.pi / 4 <= longitude <= 3*np.pi/4:
            return 2
        elif 3*np.pi / 4 <= longitude or longitude <= -3*np.pi/4:
            return 3
        elif -3*np.pi/4 <= longitude <= -np.pi/4:
            return 4
    # above a certain latitude, in panel 5
    elif latitude > np.pi / 4:
        return 5
    # below a certain latitude, in panel 6
    elif latitude < -np.pi/4:
        return 6


class CubedSphere(object):
    def __init__(self, proj_angle, vert_angles):
        # convert degrees to radians by multiplying by a factor of pi/180
        self.proj_angle = proj_angle * np.pi / 180
        self.vert_angles = vert_angles * np.pi / 180

        print(self.proj_angle)
        print(self.vert_angles)


def main():
    ds = load_data(data_folder='ARI', load_function=ARI, domain='magnitude_db', side='left')
    print(len(ds))
    for angle in ds._selected_angles.keys():
        if angle == -170.0:
            CubedSphere(proj_angle=angle, vert_angles=ds._selected_angles[angle])


if __name__ == '__main__':
    main()
