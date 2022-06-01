import cmath
import pickle

import numpy as np
import scipy
import torch

from barycentric_calcs import calc_barycentric_coordinates, calc_all_distances
from convert_coordinates import convert_cube_to_sphere, convert_sphere_to_cartesian

PI_4 = np.pi / 4


def generate_euclidean_cube(measured_coords, filename, edge_len=24):
    cube_coords, sphere_coords = [], []
    for panel in range(1, 6):
        for x in np.linspace(-PI_4, PI_4, edge_len, endpoint=False):
            for y in np.linspace(-PI_4, PI_4, edge_len, endpoint=False):
                x_i, y_i = x + PI_4 / edge_len, y + PI_4 / edge_len
                cube_coords.append((panel, x_i, y_i))
                sphere_coords.append(convert_cube_to_sphere(panel, x_i, y_i))

    euclidean_sphere_triangles = []
    euclidean_sphere_coeffs = []
    for p in sphere_coords:
        if p[0] is not None:
            triangle_vertices = get_triangle_vertices(elevation=p[0], azimuth=p[1],
                                                      sphere_coords=measured_coords)
            coeffs = calc_barycentric_coordinates(elevation=p[0], azimuth=p[1], closest_points=triangle_vertices)
            euclidean_sphere_triangles.append(triangle_vertices)
            euclidean_sphere_coeffs.append(coeffs)
        else:
            euclidean_sphere_triangles.append(None)
            euclidean_sphere_coeffs.append(None)

    # save euclidean_cube, euclidean_sphere, euclidean_sphere_triangles, euclidean_sphere_coeffs
    with open(filename, "wb") as file:
        pickle.dump((cube_coords, sphere_coords, euclidean_sphere_triangles, euclidean_sphere_coeffs), file)


def save_euclidean_cube(edge_len=24):
    sphere_coords = []
    for panel in range(1, 6):
        for x in np.linspace(-PI_4, PI_4, edge_len, endpoint=False):
            for y in np.linspace(-PI_4, PI_4, edge_len, endpoint=False):
                x_i, y_i = x + PI_4 / edge_len, y + PI_4 / edge_len
                sphere_coords.append(convert_cube_to_sphere(panel, x_i, y_i))
    with open('generated_coordinates.txt', 'w') as f:
        for coord in sphere_coords:
            print(coord)
            f.write(str(coord[0]))
            f.write(", ")
            f.write(str(coord[1]))
            f.write('\n')


def get_feature_for_point(elevation, azimuth, all_coords, subject_features):
    all_coords_row = all_coords.query(f'elevation == {elevation} & azimuth == {azimuth}')
    azimuth_index = int(all_coords_row.azimuth_index)
    elevation_index = int(all_coords_row.elevation_index)
    return subject_features[azimuth_index][elevation_index]


def get_possible_triangles(max_vertex_index, point_distances):
    possible_triangles = []
    for v0 in range(max_vertex_index - 1):
        for v1 in range(v0 + 1, max_vertex_index):
            for v2 in range(v1 + 1, max_vertex_index + 1):
                total_dist = point_distances[v0][2] + point_distances[v1][2] + point_distances[v2][2]
                possible_triangles.append((v0, v1, v2, total_dist))

    return sorted(possible_triangles, key=lambda x: x[3])


def triangle_encloses_point(elevation, azimuth, triangle_coordinates):
    # convert point of interest to cartesian coordinates and add to array
    x, y, z, _ = convert_sphere_to_cartesian([[elevation, azimuth]])
    point = np.array([x, y, z])
    # convert triangle coordinates to cartesian and add to array
    x_triangle, y_triangle, z_triangle, _ = convert_sphere_to_cartesian(triangle_coordinates)
    triangle_points = np.array([x_triangle, y_triangle, z_triangle])

    # check if matrix is singular
    if np.linalg.matrix_rank(triangle_points) < 3:
        return False

    # solve system of equations
    solution = np.linalg.solve(triangle_points, point)

    # this checks that a point lies in a spherical triangle by checking that the vector formed from the center of the
    # sphere to the point of interest intersects the plane formed by the triangle's vertices
    # check that constraints are satisfied
    solution_sum = np.sum(solution)
    solution_lambda = 1. / solution_sum

    return solution_lambda > 0 and np.all(solution > 0)


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
        # possible_triangles = get_possible_triangles(len(point_distances) - 1, point_distances)
        possible_triangles = get_possible_triangles(400, point_distances)
        for v0, v1, v2, _ in possible_triangles:
            triangle_vertices = [point_distances[v0][:2], point_distances[v1][:2], point_distances[v2][:2]]

            # for each triangle, check if it encloses the point
            if triangle_encloses_point(elevation, azimuth, triangle_vertices):
                selected_triangle_vertices = triangle_vertices
                if v2 > 300:
                    print(f'\nselected vertices for {round(elevation, 2), round(azimuth, 2)}: {v0, v1, v2}')
                    print(elevation, azimuth)
                    print(selected_triangle_vertices)
                break

    # if no triangles enclose the point, this will return none
    return selected_triangle_vertices


def calc_interpolated_feature(triangle_vertices, coeffs, all_coords, subject_features):
    # get features for each of the three closest points, add to a list in order of closest to farthest
    features = []
    for p in triangle_vertices:
        features_p = get_feature_for_point(p[0], p[1], all_coords, subject_features)
        features.append(features_p)

    # based on equation 6 in "3D Tune-In Toolkit: An open-source library for real-time binaural spatialisation"
    interpolated_feature = coeffs["alpha"] * features[0] + coeffs["beta"] * features[1] + coeffs["gamma"] * features[2]

    return interpolated_feature


def calc_all_interpolated_features(cs, features, euclidean_sphere, euclidean_sphere_triangles, euclidean_sphere_coeffs):
    selected_feature_interpolated = []
    for i, p in enumerate(euclidean_sphere):
        if p[0] is not None:
            features_p = calc_interpolated_feature(triangle_vertices=euclidean_sphere_triangles[i],
                                                   coeffs=euclidean_sphere_coeffs[i],
                                                   all_coords=cs.get_all_coords(),
                                                   subject_features=features)

            selected_feature_interpolated.append(features_p)
        else:
            selected_feature_interpolated.append(None)

    return selected_feature_interpolated


def calc_hrtf(hrirs):
    magnitudes = []
    phases = []
    for hrir in hrirs:
        hrtf = scipy.fft.fft(hrir)
        magnitude = abs(hrtf)
        phase = [cmath.phase(x) for x in hrtf]
        magnitudes.append(magnitude)
        phases.append(phase)

    return magnitudes, phases


def rows_to_cols(rows, edge_len, pad_width):
    top_edge = []
    for i in range(edge_len):
        col = []
        for j in range(pad_width):
            col.append(rows[j][i])
        top_edge.append(col)
    return top_edge


def cols_to_rows(cols, pad_width):
    edge = []
    for i in range(pad_width):
        row = [x[i] for x in cols]
        edge.append(row)
    return edge


def pad_column(column, pad_width):
    padded = []
    for col in column:
        col_pad = pad_width * [col[0]] + col + pad_width * [col[-1]]
        padded.append(col_pad)
    return padded


def create_edge_dict(magnitudes, pad_width):
    panel_edges = []
    for panel in range(5):
        left = magnitudes[panel][:pad_width]  # get left column(s) (all lowest x)
        right = magnitudes[panel][-pad_width:]  # get right column(s) (all highest x)
        bottom = [x[:pad_width] for x in magnitudes[panel]]  # get bottom row(s) (all lowest y)
        top = [x[-pad_width:] for x in magnitudes[panel]]  # get top row(s) (all highest y)
        edge_dict = {'left': left, 'right': right, 'top': top, 'bottom': bottom}
        panel_edges.append(edge_dict)
    return panel_edges


def pad_equatorial_panel(magnitudes_panel, panel, panel_edges, edge_len, pad_width):
    # get row/column from panel 4 to pad top edge
    if panel == 0:
        # no need to reverse at all
        top_edge = panel_edges[4]['bottom'].copy()
    elif panel == 1:
        # need to reverse in only one direction
        top_edge = rows_to_cols(panel_edges[4]['right'].copy(), edge_len=edge_len, pad_width=pad_width)
        top_edge = [list(reversed(col)) for col in top_edge]
    elif panel == 2:
        # need to reverse in both directions
        top_edge = panel_edges[4]['top'].copy()
        top_edge.reverse()
        top_edge = [list(reversed(col)) for col in top_edge]
    else:
        # need to reverse in only one direction
        top_edge = rows_to_cols(panel_edges[4]['left'].copy(), edge_len=edge_len, pad_width=pad_width)
        top_edge.reverse()

    # pad TOP AND BOTTOM of panel on a column by column basis
    # pad bottom of column via replication
    column_list = []
    for i in range(edge_len):
        column = magnitudes_panel[i]
        col_pad = pad_width * [column[0]] + column + top_edge[i]
        column_list.append(col_pad)

    # pad LEFT AND RIGHT side of each panel around horizontal plane
    # get panel index for left and right panel
    left_panel = (panel - 1) % 4
    right_panel = (panel + 1) % 4

    # get the rightmost column of the left panel, and pad top and bottom with edge values
    left_col = panel_edges[left_panel]['right']
    left_col_pad = pad_column(left_col, pad_width)

    # get the leftmost column of the right panel, and pad top and bottom with edge values
    right_col = panel_edges[right_panel]['left']
    right_col_pad = pad_column(right_col, pad_width)

    # COMBINE left column, padded center columns, and right column to get final version
    return left_col_pad + column_list + right_col_pad


def pad_top_panel(magnitudes_top, panel_edges, edge_len, pad_width):
    # pad TOP AND BOTTOM of panel on a column by column basis
    column_list = []
    bottom_edge = panel_edges[0]['top'].copy()
    top_edge = panel_edges[2]['top'].copy()
    top_edge.reverse()
    top_edge = [list(reversed(col)) for col in top_edge]
    for i in range(edge_len):
        column = magnitudes_top[i]
        col_pad = bottom_edge[i] + column + top_edge[i]
        column_list.append(col_pad)

    # get the top row of panel 3, reverse it, and pad top and bottom with edge values
    left_col = panel_edges[3]['top'].copy()
    left_col.reverse()
    left_col_pad = pad_width * [left_col[0]] + left_col + pad_width * [left_col[-1]]

    # get the top row of panel 1, and pad top and bottom with edge values
    right_col = panel_edges[1]['top'].copy()
    right_col_pad = pad_width * [right_col[0]] + right_col + pad_width * [right_col[-1]]
    right_col_pad = [list(reversed(col)) for col in right_col_pad]

    # convert from columns to rows
    left_col_pad = cols_to_rows(left_col_pad, pad_width)
    right_col_pad = cols_to_rows(right_col_pad, pad_width)

    # COMBINE left column, padded center columns, and right column to get final version
    return left_col_pad + column_list + right_col_pad


def pad_cubed_sphere(magnitudes, pad_width):
    edge_len = len(magnitudes[0])

    # create a list of dictionaries (one for each panel) containing the left, right, top and bottom edges for each panel
    panel_edges = create_edge_dict(magnitudes, pad_width)

    # create empty list of lists of lists
    magnitudes_pad = [[[[] for _ in range(edge_len + 2 * pad_width)] for _ in range(edge_len + 2 * pad_width)]
                      for _ in range(5)]

    # diagram of unfolded cube, with panel indices
    #             _______
    #            |       |
    #            |   4   |
    #     _______|_______|_______ _______
    #    |       |       |       |       |
    #    |   3   |   0   |   1   |   2   |
    #    |_______|_______|_______|_______|
    # In all cases, low values of x and y are situated in lower left of the unfolded sphere

    # pad the 4 panels around the horizontal plane
    for panel in range(4):
        magnitudes_pad[panel] = pad_equatorial_panel(magnitudes[panel], panel, panel_edges, edge_len, pad_width)

    # now pad for the top panel (panel 4)
    magnitudes_pad[4] = pad_top_panel(magnitudes[4], panel_edges, edge_len, pad_width)

    return magnitudes_pad


def interpolate_fft_pad(cs, ds, load_sphere, load_sphere_triangles, load_sphere_coeffs, load_cube, edge_len, pad_width):
    # interpolated_hrirs is a list of interpolated HRIRs corresponding to the points specified in load_sphere and
    # load_cube, all three lists share the same ordering
    interpolated_hrirs = calc_all_interpolated_features(cs, ds['features'],
                                                        load_sphere, load_sphere_triangles, load_sphere_coeffs)
    magnitudes, phases = calc_hrtf(interpolated_hrirs)

    # create empty list of lists of lists and initialize counter
    magnitudes_raw = [[[[] for _ in range(edge_len)] for _ in range(edge_len)] for _ in range(5)]
    count = 0

    for panel, x, y in load_cube:
        # based on cube coordinates, get indices for magnitudes list of lists
        i = panel - 1
        j = round(edge_len * (x - (PI_4 / edge_len) + PI_4) / (np.pi / 2))
        k = round(edge_len * (y - (PI_4 / edge_len) + PI_4) / (np.pi / 2))

        # add to list of lists of lists and increment counter
        magnitudes_raw[i][j][k] = magnitudes[count]
        count += 1

    # pad each panel of the cubed sphere appropriately
    magnitudes_pad = pad_cubed_sphere(magnitudes_raw, pad_width)

    # convert list of numpy arrays into a single array, such that converting into tensor is faster
    return torch.tensor(np.array(magnitudes_pad))
