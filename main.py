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
from utils import get_feature_for_point, generate_euclidean_cube, triangle_encloses_point, get_possible_triangles, \
    calc_all_interpolated_features

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

    # if no triangles enclose the point, this will return none
    return selected_triangle_vertices


def main():
    ds: ARI = load_data(data_folder='ARI', load_function=ARI, domain='time', side='left')

    # need to use protected member to get this data, no getters
    cs = CubedSphere(sphere_coords=ds._selected_angles)

    euclidean_cube, euclidean_sphere = generate_euclidean_cube(edge_len=24)

    euclidean_sphere_triangles = []
    euclidean_sphere_coeffs = []
    for p in euclidean_sphere:
        if p[0] is not None:
            triangle_vertices = get_triangle_vertices(elevation=p[0], azimuth=p[1],
                                                      sphere_coords=cs.get_sphere_coords())
            coeffs = calc_barycentric_coordinates(elevation=p[0], azimuth=p[1], closest_points=triangle_vertices)
            euclidean_sphere_triangles.append(triangle_vertices)
            euclidean_sphere_coeffs.append(coeffs)
        else:
            euclidean_sphere_triangles.append(None)
            euclidean_sphere_coeffs.append(None)

    plot_interpolated_features(cs, ds[0]['features'], 20, euclidean_cube, euclidean_sphere,
                               euclidean_sphere_triangles, euclidean_sphere_coeffs)


if __name__ == '__main__':
    main()
