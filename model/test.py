import os

import scipy
import torch

from model.model import Generator
from model.util import spectral_distortion_metric
from plot import plot_magnitude_spectrums


def test(config, val_prefetcher):
    # source: https://github.com/Lornatang/SRGAN-PyTorch/blob/main/test.py
    # Initialize super-resolution model
    ngpu = config.ngpu
    path = config.path
    device = torch.device(config.device_name if (
            torch.cuda.is_available() and ngpu > 0) else "cpu")
    model = Generator().to(device=device)
    print("Build SRGAN model successfully.")

    # Load super-resolution model weights
    model.load_state_dict(torch.load(f"{config.model_path}/Gen.pt"))
    print(f"Load SRGAN model weights `{os.path.abspath(config.model_path)}` successfully.")

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_param_mb = param_size / 1024 ** 2
    size_buffer_mb = buffer_size / 1024 ** 2
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('param size: {:.3f}MB'.format(size_param_mb))
    print('buffer size: {:.3f}MB'.format(size_buffer_mb))
    print('model size: {:.3f}MB'.format(size_all_mb))

    # get list of positive frequencies of HRTF for plotting magnitude spectrum
    hrir_samplerate = 48000.0
    all_freqs = scipy.fft.fftfreq(256, 1 / hrir_samplerate)
    pos_freqs = all_freqs[all_freqs >= 0]

    # Start the verification mode of the model.
    model.eval()

    # Initialize the data loader and load the first batch of data
    val_prefetcher.reset()
    batch_data = val_prefetcher.next()

    val_loss = 0.
    while batch_data is not None:
        # Transfer in-memory data to CUDA devices to speed up validation
        lr = batch_data["lr"].to(device=device, memory_format=torch.contiguous_format,
                                 non_blocking=True, dtype=torch.float)
        hr = batch_data["hr"].to(device=device, memory_format=torch.contiguous_format,
                                 non_blocking=True, dtype=torch.float)
        filename = batch_data["filename"]

        # Use the generator model to generate fake samples
        with torch.no_grad():
            sr = model(lr)

        val_loss += spectral_distortion_metric(sr, hr).item()

        # Preload the next batch of data
        batch_data = val_prefetcher.next()

    avg_loss = val_loss / len(val_prefetcher)
    print(f"Average validation spectral distortion metric: {avg_loss}")

    # create magnitude spectrum plot
    magnitudes_real = torch.permute(hr.detach().cpu()[0], (1, 2, 3, 0))
    magnitudes_interpolated = torch.permute(sr.detach().cpu()[0], (1, 2, 3, 0))

    if len(filename) != 1:
        print("Filename contains multiple values, including:")
        print(filename)

    if filename[0][-5:] == 'right':
        ear_label = 'right'
    elif filename[0][-4:] == 'left':
        ear_label = 'left'
    else:
        ear_label = 'unknown'

    plot_magnitude_spectrums(pos_freqs, magnitudes_real, magnitudes_interpolated,
                             ear_label, "val", path, log_scale_magnitudes=True)
    # TODO: might be worthwhile to plot for every validation HRTF
