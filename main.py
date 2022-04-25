from hrtfdata.torch.planar import CHEDARPlane, ARIPlane
from hrtfdata.torch import collate_dict_dataset
from torch.utils.data import DataLoader
from pathlib import Path


def load_data(data_folder, load_function, plane, domain, side):
    base_dir = Path('/Users/madsjenkins/Imperial/HRTF/Volumes/home/HRTF Datasets')
    return load_function(base_dir / data_folder, plane, domain=domain, side=side)


if __name__ == '__main__':
    ds = load_data(data_folder='ARI', load_function=ARIPlane,
                   plane='median', domain='magnitude_db', side='left')
    DataLoader(ds, collate_fn=collate_dict_dataset)
    print(len(ds))
