import cmath
import pickle

import numpy as np
import scipy

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


def pad_cubed_sphere(magnitudes):
    edge_len = len(magnitudes[0])
    # create a list of dictionaries (one for each panel) containing the left, right, top and bottom edges for each panel
    panel_edges = []
    for panel in range(5):
        left = magnitudes[panel][0]  # get left column (all lowest x)
        right = magnitudes[panel][-1]  # get right column (all highest x)
        bottom = [x[0] for x in magnitudes[panel]]  # get bottom row (all lowest y)
        top = [x[-1] for x in magnitudes[panel]]  # get bottom row (all highest y)
        edge_dict = {'left': left, 'right': right, 'top': top, 'bottom': bottom}
        panel_edges.append(edge_dict)

    magnitudes_pad = [[[[] for _ in range(edge_len + 2)] for _ in range(edge_len + 2)] for _ in range(5)]
    # pad the 4 panels around the horizontal plane
    for panel in range(4):
        # get row/column from panel 4 to pad top edge
        #             _______
        #            |       |
        #            |   4   |
        #     _______|_______|_______ _______
        #    |       |       |       |       |
        #    |   3   |   0   |   1   |   2   |
        #    |_______|_______|_______|_______|
        # In all cases, low values of x and y are situated in lower left of the unfolded sphere

        if panel == 0:
            top_edge = panel_edges[4]['bottom'].copy()
        elif panel == 1:
            top_edge = panel_edges[4]['right'].copy()
        elif panel == 2:
            # need to reverse
            top_edge = panel_edges[4]['top'].copy()
            top_edge.reverse()
        else:
            # need to reverse
            top_edge = panel_edges[4]['left'].copy()
            top_edge.reverse()

        # pad TOP AND BOTTOM of panel on a column by column basis
        # pad bottom of column via replication
        column_list = []
        for i in range(edge_len):
            column = magnitudes[panel][i]
            col_pad = [column[0]] + column + [top_edge[i]]
            column_list.append(col_pad)

        # pad LEFT AND RIGHT side of each panel around horizontal plane
        # get panel index for left and right panel
        left_panel = (panel - 1) % 4
        right_panel = (panel + 1) % 4
        # get the rightmost column of the left panel, and pad top and bottom with edge values
        left_col = panel_edges[left_panel]['right']
        left_col_pad = [[left_col[0]] + left_col + [left_col[-1]]]
        # get the leftmost column of the right panel, and pad top and bottom with edge values
        right_col = panel_edges[right_panel]['left']
        right_col_pad = [[right_col[0]] + right_col + [right_col[-1]]]

        # COMBINE left column, padded center columns, and right column to get final version
        magnitudes_pad[panel] = left_col_pad + column_list + right_col_pad

    # now pad for the top panel (panel 4)
    column_list = []
    bottom_edge = panel_edges[0]['top'].copy()
    top_edge = panel_edges[2]['top'].copy()
    top_edge.reverse()
    for i in range(edge_len):
        column = magnitudes[4][i]
        col_pad = [bottom_edge[i]] + column + [top_edge[i]]
        column_list.append(col_pad)

    # get the top row of panel 3, reverse it, and pad top and bottom with edge values
    left_col = panel_edges[3]['top'].copy()
    left_col.reverse()
    left_col_pad = [[left_col[0]] + left_col + [left_col[-1]]]
    # get the top row of panel 1, and pad top and bottom with edge values
    right_col = panel_edges[1]['top'].copy()
    right_col_pad = [[right_col[0]] + right_col + [right_col[-1]]]

    # COMBINE left column, padded center columns, and right column to get final version
    magnitudes_pad[4] = left_col_pad + column_list + right_col_pad

    return(magnitudes_pad)
