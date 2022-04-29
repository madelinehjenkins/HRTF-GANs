from hrtfdata.torch.full import ARI
from hrtfdata.torch import collate_dict_dataset
from torch.utils.data import DataLoader
from pathlib import Path


def load_data(data_folder, load_function, domain, side):
    base_dir = Path('/Users/madsjenkins/Imperial/HRTF/Volumes/home/HRTF Datasets')
    return load_function(base_dir / data_folder,
                         feature_spec={"hrirs": {'side': side, 'domain': domain}},
                         target_spec={"side": {}},
                         group_spec={"subject": {}})


def main():
    ds = load_data(data_folder='ARI', load_function=ARI, domain='magnitude_db', side='left')
    # DataLoader(ds, collate_fn=collate_dict_dataset)
    print(len(ds))


if __name__ == '__main__':
    main()
