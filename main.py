import itertools

import pandas as pd
import torch
from hrtfdata.torch.full import ARI, CHEDAR
from hrtfdata.torch import collate_dict_dataset
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

from matplotlib import cm
from collections import Counter

from barycentric_calcs import calc_all_distances, calc_barycentric_coordinates
from cubed_sphere import CubedSphere
from plot import plot_3d_shape, plot_flat_cube, plot_impulse_response
from convert_coordinates import convert_cube_to_sphere, convert_sphere_to_cube, convert_sphere_to_cartesian, \
    convert_cube_to_cartesian
from utils import get_feature_for_point, generate_euclidean_cube, triangle_encloses_point, get_possible_triangles

PI_4 = np.pi / 4


def load_data(data_folder, load_function, domain, side):
    base_dir = Path('/Users/madsjenkins/Imperial/HRTF/Volumes/home/HRTF Datasets')
    return load_function(base_dir / data_folder,
                         feature_spec={"hrirs": {'side': side, 'domain': domain}},
                         target_spec={"side": {}},
                         group_spec={"subject": {}},
                         subject_ids="first")  # temporary measure to avoid loading entire dataset each time


def get_triangle_vertices(elevation, azimuth, sphere_coords):
    selected_triangle_vertices = None
    # get distances from point of interest to every other point
    point_distances = calc_all_distances(elevation=elevation, azimuth=azimuth, sphere_coords=sphere_coords)

    # first try triangle formed by closest points
    triangle_vertices = [point_distances[0][:2], point_distances[1][:2], point_distances[2][:2]]
    if triangle_encloses_point(elevation, azimuth, triangle_vertices):
        selected_triangle_vertices = triangle_vertices
    else:
        # failing that, examine all possible triangles
        # possible triangles is sorted from shortest total distance to longest total distance
        possible_triangles = get_possible_triangles(300, point_distances)
        # TODO: including all vertices is more correct but runs very slowly
        # possible_triangles = get_possible_triangles(len(point_distances) - 1, point_distances)
        for v0, v1, v2, _ in possible_triangles:
            triangle_vertices = [point_distances[v0][:2], point_distances[v1][:2], point_distances[v2][:2]]

            # for each triangle, check if it encloses the point
            if triangle_encloses_point(elevation, azimuth, triangle_vertices):
                selected_triangle_vertices = triangle_vertices
                if v2 > 10:
                    print(f'\nselected vertices for {round(elevation, 2), round(azimuth, 2)}: {v0, v1, v2}')
                    print(elevation, azimuth)
                    print(selected_triangle_vertices)
                break

    # if no triangles enclose the point, return none (hopefully this never happens)
    if selected_triangle_vertices is None:
        print('it happened')
        print(f'elevation, azimuth: {round(elevation, 2), round(azimuth, 2)}')
    return selected_triangle_vertices


def calc_interpolated_feature(elevation, azimuth, sphere_coords, all_coords, subject_features):
    triangle_vertices = get_triangle_vertices(elevation=elevation, azimuth=azimuth, sphere_coords=sphere_coords)
    coeffs = calc_barycentric_coordinates(elevation=elevation, azimuth=azimuth, closest_points=triangle_vertices)

    # get features for each of the three closest points, add to a list in order of closest to farthest
    features = []
    for p in triangle_vertices:
        features_p = get_feature_for_point(p[0], p[1], all_coords, subject_features)
        features.append(features_p)

    # based on equation 6 in "3D Tune-In Toolkit: An open-source library for real-time binaural spatialisation"
    interpolated_feature = coeffs["alpha"] * features[0] + coeffs["beta"] * features[1] + coeffs["gamma"] * features[2]

    return interpolated_feature


def create_interpolated_plots(cs, features, feature_index, edge_len=24):
    euclidean_cube, euclidean_sphere = generate_euclidean_cube(edge_len=edge_len)
    all_coords = cs.get_all_coords()

    selected_feature_raw = []
    for p in cs.get_sphere_coords():
        if p[0] is not None:
            features_p = get_feature_for_point(p[0], p[1], all_coords, features)
            selected_feature_raw.append(features_p[feature_index])
        else:
            selected_feature_raw.append(None)

    plot_3d_shape("sphere", cs.get_sphere_coords(), shading=selected_feature_raw)
    plot_3d_shape("cube", cs.get_cube_coords(), shading=selected_feature_raw)
    plot_flat_cube(cs.get_cube_coords(), shading=selected_feature_raw)

    selected_feature_interpolated = []
    for p in euclidean_sphere:
        if p[0] is not None:
            features_p = calc_interpolated_feature(elevation=p[0], azimuth=p[1],
                                                   sphere_coords=cs.get_sphere_coords(), all_coords=all_coords,
                                                   subject_features=features)
            selected_feature_interpolated.append(features_p[feature_index])
        else:
            selected_feature_interpolated.append(None)

    plot_flat_cube(euclidean_cube, shading=selected_feature_interpolated)
    plot_3d_shape("cube", euclidean_cube, shading=selected_feature_interpolated)
    plot_3d_shape("sphere", euclidean_sphere, shading=selected_feature_interpolated)


def main():
    ds: ARI = load_data(data_folder='ARI', load_function=ARI, domain='time', side='left')

    # need to use protected member to get this data, no getters
    cs = CubedSphere(sphere_coords=ds._selected_angles)

    create_interpolated_plots(cs, ds[0]['features'], 20, edge_len=2)


if __name__ == '__main__':
    main()
