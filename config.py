import json
from pathlib import Path


class Config:
    """Config class

    Set using HPC to true in order to use appropriate paths for HPC
    """
    def __init__(self, tag, using_hpc):
        self.tag = tag
        self.path = f'runs/{self.tag}'

        # data dirs
        if using_hpc:
            self.raw_hrtf_dir = Path('../project/sonicom/live/HRTF Datasets')
        else:
            self.raw_hrtf_dir = Path('/Users/madsjenkins/Imperial/HRTF/Volumes/home/HRTF Datasets')
        self.train_hrtf_dir = 'projected_data/train'
        self.valid_hrtf_dir = 'projected_data/valid'

        # Data processing parameters
        self.pad_width = 2
        self.hrtf_size = 16
        self.upscale_factor = 4
        self.train_samples_ratio = 0.8

        # Training hyperparams
        self.batch_size = 32
        self.num_workers = 4
        self.num_epochs = 2  # was originally 250
        self.lr_gen = 0.0001
        self.lr_dis = 0.0001
        # how often to train the generator
        self.critic_iters = 4

        # betas for Adam optimizer
        self.beta1 = 0.9
        self.beta2 = 0.999

        self.ngpu = 1
        if self.ngpu > 0:
            self.device_name = "cuda:0"
        else:
            self.device_name = 'cpu'

    def save(self):
        j = {}
        for k, v in self.__dict__.items():
            j[k] = v
        with open(f'{self.path}/config.json', 'w') as f:
            json.dump(j, f)

    def load(self):
        with open(f'{self.path}/config.json', 'r') as f:
            j = json.load(f)
            for k, v in j.items():
                setattr(self, k, v)
    
    def get_train_params(self):
        return self.batch_size, self.beta1, self.beta2, self.num_epochs, self.lr_gen, self.lr_dis, self.critic_iters