from hrtfdata.torch.full import ARI
from hrtfdata.torch import collate_dict_dataset
from torch.utils.data import DataLoader
from pathlib import Path


def load_data(data_folder, load_function):
    base_dir = Path('/Users/madsjenkins/Imperial/HRTF/Volumes/home/HRTF Datasets')
    return load_function(base_dir / data_folder)


def main():
    ds = load_data(data_folder='ARI', load_function=ARI)
    # DataLoader(ds, collate_fn=collate_dict_dataset)
    print(len(ds))


if __name__ == '__main__':
    main()
