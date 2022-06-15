import json

import torch


class Config:
    """Config class
    """
    def __init__(self, tag):
        self.tag = tag
        self.path = f'runs/{self.tag}'
        # data dirs
        self.data_path = 'data/Example_ppp.png'
        self.train_hrtf_dir = 'projected_data/train'
        self.valid_hrtf_dir = 'projected_data/valid'
        self.hrtf_size = 16

        self.upscale_factor = 4
        self.net_type = 'gan'
        self.l = 64
        self.n_phases = 2
        # Training hyperparams
        self.batch_size = 64
        self.num_workers = 4
        # betas for Adam optimizer
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.num_epochs = 5 # was originally 250
        self.iters = 1000
        self.lr_gen = 0.0001
        self.lr_dis = 0.0001
        self.Lambda = 10
        self.critic_iters = 10
        self.lz = 4
        self.lf = 4
        self.ngpu = 1
        if self.ngpu > 0:
            self.device_name = "cuda:0"
        else:
            self.device_name = 'cpu'
        self.nz = 100
        # Architecture
        self.lays = 4
        self.laysd = 5

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
        return self.l, self.batch_size, self.beta1, self.beta2, self.num_epochs, self.iters, self.lr_gen, self.lr_dis, self.Lambda, self.critic_iters, self.lz, self.nz


