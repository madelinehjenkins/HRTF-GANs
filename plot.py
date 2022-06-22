import matplotlib.pyplot as plt
import itertools

import numpy as np
import torch
from matplotlib import patches
from matplotlib.ticker import LinearLocator

from preprocessing.convert_coordinates import convert_sphere_to_cartesian, convert_cube_to_cartesian
from preprocessing.utils import calc_all_interpolated_features, get_feature_for_point

PI_4 = np.pi / 4


def plot_3d_shape(shape, coordinates, shading=None):
    """Plot points from a sphere or a cubed sphere in 3D

    :param shape: either "sphere" or "cube" to specify the shape to plot
    :param coordinates: A list of coordinates to plot, either (elevation, azimuth) for spheres or
                        (panel, x, y) for cubed spheres
    :param shading: A list of values equal in length to the number of coordinates that is used for shading the points
    """
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Format data.
    if shape == "sphere":
        x, y, z, mask = convert_sphere_to_cartesian(coordinates)
    elif shape == "cube":
        x, y, z, mask = convert_cube_to_cartesian(coordinates)
    else:
        raise RuntimeError("Please provide a valid shape, either 'sphere' or 'cube'.")

    if shading is not None:
        shading = list(itertools.compress(shading, mask))

    # Plot the surface.
    sc = ax.scatter(x, y, z, c=shading, s=10,
                    linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    plt.colorbar(sc)

    plt.show()


def plot_flat_cube(cube_coords, shading=None):
    """Plot points from cubed sphere in its flattened form

    :param cube_coords: A list of coordinates to plot in the form (panel, x, y) for cubed spheres
    :param shading: A list of values equal in length to the number of coordinates that is used for shading the points
    """
    fig, ax = plt.subplots()

    # Format data.
    x, y = [], []
    mask = []

    for panel, p, q in cube_coords:
        if not np.isnan(p) and not np.isnan(q):
            mask.append(True)

            if panel == 1:
                x_i, y_i = p, q
            elif panel == 2:
                x_i, y_i = p + np.pi / 2, q
            elif panel == 3:
                x_i, y_i = p + np.pi, q
            elif panel == 4:
                x_i, y_i = p - np.pi / 2, q
            elif panel == 5:
                x_i, y_i = p, q + np.pi / 2
            else:
                x_i, y_i = p, q - np.pi / 2

            x.append(x_i)
            y.append(y_i)
        else:
            mask.append(False)

    x, y = np.asarray(x), np.asarray(y)

    if shading is not None:
        shading = list(itertools.compress(shading, mask))

    # draw lines outlining cube
    ax.hlines(y=-PI_4, xmin=-3 * PI_4, xmax=5 * PI_4, linewidth=2, color="grey")
    ax.hlines(y=PI_4, xmin=-3 * PI_4, xmax=5 * PI_4, linewidth=2, color="grey")
    ax.hlines(y=3 * PI_4, xmin=-PI_4, xmax=PI_4, linewidth=2, color="grey")

    ax.vlines(x=-3 * PI_4, ymin=-PI_4, ymax=PI_4, linewidth=2, color="grey")
    ax.vlines(x=-PI_4, ymin=-PI_4, ymax=3 * PI_4, linewidth=2, color="grey")
    ax.vlines(x=PI_4, ymin=-PI_4, ymax=3 * PI_4, linewidth=2, color="grey")
    ax.vlines(x=3 * PI_4, ymin=-PI_4, ymax=PI_4, linewidth=2, color="grey")
    ax.vlines(x=5 * PI_4, ymin=-PI_4, ymax=PI_4, linewidth=2, color="grey")

    # Plot the surface.
    sc = ax.scatter(x, y, c=shading, s=10,
                    linewidth=0, antialiased=False)
    plt.colorbar(sc)

    fig.tight_layout()
    fig.set_size_inches(9, 4)
    plt.show()


def plot_impulse_response(times, title=""):
    """Plot a single impulse response, where sound pressure levels are provided as a list"""
    plt.plot(times)
    plt.title(title, fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Sound Pressure Level", fontsize=14)
    plt.show()


def plot_ir_subplots(hrir1, hrir2, title1="", title2="", suptitle=""):
    """Plot two IRs as subplots"""
    fig, axs = plt.subplots(2)
    fig.suptitle(suptitle, fontsize=16)
    axs[0].plot(hrir1)
    axs[0].set_xlabel('Time (samples)', fontsize=14)
    axs[0].set_title(title1, fontsize=16)
    axs[1].plot(hrir2)
    axs[1].set_xlabel('Time (samples)', fontsize=14)
    axs[1].set_title(title2, fontsize=16)
    fig.supylabel("Sound Pressure Level", fontsize=14)
    plt.subplots_adjust(left=0.15, right=0.95, hspace=0.7, top=0.85)
    plt.show()


def plot_interpolated_features(cs, features, i, euclidean_cube, euclidean_sphere, sphere_triangles, sphere_coeffs):
    """Plot i-th interpolated feature on flatted cubed sphere, 3D cubed sphere, & 3D sphere"""
    # TODO: test this function to ensure I have not broken it

    interpolated = calc_all_interpolated_features(cs, features, euclidean_sphere, sphere_triangles, sphere_coeffs)

    plot_flat_cube(euclidean_cube, shading=interpolated[i])
    plot_3d_shape("cube", euclidean_cube, shading=interpolated[i])
    plot_3d_shape("sphere", euclidean_sphere, shading=interpolated[i])


def plot_original_features(cs, features, i):
    """Plot i-th original feature on flatted cubed sphere, 3D cubed sphere, & 3D sphere"""

    selected_feature_raw = []
    for p in cs.get_sphere_coords():
        if p[0] is not None:
            features_p = get_feature_for_point(p[0], p[1], cs.get_all_coords(), features)
            selected_feature_raw.append(features_p[i])
        else:
            selected_feature_raw.append(None)

    plot_3d_shape("sphere", cs.get_sphere_coords(), shading=selected_feature_raw)
    plot_3d_shape("cube", cs.get_cube_coords(), shading=selected_feature_raw)
    plot_flat_cube(cs.get_cube_coords(), shading=selected_feature_raw)


def plot_padded_panels(panel_tensors, edge_len, pad_width, label_cells, title):
    """Plot panels with padding, indicating on the plot which areas are padded vs. not

    Useful for verifying that padding has been performed correctly
    """

    # panel tensor must be of shape (5, n, n) where n = edge_len + padding
    fig, axs = plt.subplots(2, 4, sharex='col', sharey='row', figsize=(9, 5))

    plot_locs = [(1, 1), (1, 2), (1, 3), (1, 0), (0, 1)]
    for panel in range(5):
        row, col = plot_locs[panel]
        plot_tensor = torch.flip(panel_tensors[panel].T, [0])
        axs[row, col].imshow(plot_tensor, vmin=torch.min(panel_tensors), vmax=torch.max(panel_tensors))

        # Create a Rectangle patch to outline panel and separate padded area
        rect = patches.Rectangle((0.5 + (pad_width - 1), 0.5 + (pad_width - 1)), edge_len, edge_len,
                                 linewidth=1, edgecolor='white', facecolor='none', hatch='/')
        # Add the patch to the Axes
        axs[row, col].add_patch(rect)

        if label_cells:
            for i in range(edge_len + 2 * pad_width):
                for j in range(edge_len + 2 * pad_width):
                    axs[row, col].text(j, i, round(1000 * plot_tensor[i][j].item(), 1), ha="center", va="center",
                                       color="w")

    axs[0, 0].axis('off')
    axs[0, 2].axis('off')
    axs[0, 3].axis('off')

    # Show all ticks and label them with the respective list entries
    axs[1, 0].set_xticks([])
    axs[1, 1].set_xticks([])
    axs[1, 2].set_xticks([])
    axs[1, 3].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[1, 0].set_yticks([])

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_panel(lr, sr, hr, batch_index, epoch, path, ncol, freq_index):
    """Based on the input data to the GAN and the output from the generator, plot a single panel for the first 4 HRTFs
    in the batch, at a given freq_index
    """
    lr_selected = lr.detach().cpu()[:ncol, freq_index, :, :]
    sr_selected = sr.detach().cpu()[:ncol, freq_index, :, :]
    hr_selected = hr.detach().cpu()[:ncol, freq_index, :, :]
    min_magnitude = min((torch.min(lr_selected), torch.min(sr_selected), torch.min(hr_selected)))
    max_magnitude = max((torch.max(lr_selected), torch.max(sr_selected), torch.max(hr_selected)))

    fig, axs = plt.subplots(3, ncol, subplot_kw={'xticks': [], 'yticks': []})
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.8, hspace=0.5, wspace=0.1)

    for n, lr_hrtf in enumerate(lr_selected):
        ax = plt.subplot(3, ncol, n + 1)
        ax.imshow(lr_hrtf, vmin=min_magnitude, vmax=max_magnitude)
        ax.set_title("LR " + str(n))

    for n, sr_hrtf in enumerate(sr_selected):
        ax = plt.subplot(3, ncol, n + 1 + ncol)
        ax.imshow(sr_hrtf, vmin=min_magnitude, vmax=max_magnitude)
        ax.set_title("SR " + str(n))

    for n, hr_hrtf in enumerate(hr_selected):
        ax = plt.subplot(3, ncol, n + 1 + (2 * ncol))
        temp = ax.imshow(hr_hrtf, vmin=min_magnitude, vmax=max_magnitude)
        ax.set_title("HR " + str(n))

    fig.colorbar(temp, ax=axs, shrink=0.7)
    fig.suptitle("Comparison of LR magnitudes, their generated SR counterparts, \nand HR ground truth")

    plt.savefig(f'{path}/{epoch}_{batch_index}_slices.png')
    plt.close(fig)


def plot_losses(train_losses_d, train_losses_g, path):
    """Plot the discriminator and generator loss over time"""
    plt.figure()
    loss_d = [x.item() for x in train_losses_d]
    loss_g = [x.item() for x in train_losses_g]
    plt.plot(loss_d, label='Discriminator loss')
    plt.plot(loss_g, label='Generator loss')

    plt.title("Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'{path}/loss_curves.png')