import matplotlib.pyplot as plt
import itertools

import numpy as np
from matplotlib.ticker import LinearLocator

from convert_coordinates import convert_sphere_to_cartesian, convert_cube_to_cartesian
from utils import calc_all_interpolated_features, get_feature_for_point

PI_4 = np.pi / 4


def plot_3d_shape(shape, coordinates, shading=None):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Format data.
    if shape == "sphere":
        x, y, z, mask = convert_sphere_to_cartesian(coordinates)
    elif shape == "cube":
        x, y, z, mask = convert_cube_to_cartesian(coordinates)

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
    plt.plot(times)
    plt.title(title, fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Sound Pressure Level", fontsize=14)
    plt.show()


def plot_interpolated_features(cs, features, feature_index, euclidean_cube, euclidean_sphere,
                               euclidean_sphere_triangles, euclidean_sphere_coeffs):
    selected_feature_interpolated = calc_all_interpolated_features(cs, features, feature_index, euclidean_sphere,
                                                                   euclidean_sphere_triangles, euclidean_sphere_coeffs)

    plot_flat_cube(euclidean_cube, shading=selected_feature_interpolated)
    plot_3d_shape("cube", euclidean_cube, shading=selected_feature_interpolated)
    plot_3d_shape("sphere", euclidean_sphere, shading=selected_feature_interpolated)


def plot_original_features(cs, features, feature_index):
    selected_feature_raw = []
    for p in cs.get_sphere_coords():
        if p[0] is not None:
            features_p = get_feature_for_point(p[0], p[1], cs.get_all_coords(), features)
            selected_feature_raw.append(features_p[feature_index])
        else:
            selected_feature_raw.append(None)

    plot_3d_shape("sphere", cs.get_sphere_coords(), shading=selected_feature_raw)
    plot_3d_shape("cube", cs.get_cube_coords(), shading=selected_feature_raw)
    plot_flat_cube(cs.get_cube_coords(), shading=selected_feature_raw)