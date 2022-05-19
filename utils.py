import numpy as np

from convert_coordinates import convert_cube_to_sphere, convert_sphere_to_cartesian

PI_4 = np.pi / 4


def generate_euclidean_cube(edge_len=24):
    cube_coords, sphere_coords = [], []
    for panel in range(1, 6):
        for x in np.linspace(-PI_4, PI_4, edge_len, endpoint=False):
            for y in np.linspace(-PI_4, PI_4, edge_len, endpoint=False):
                x_i, y_i = x + PI_4 / edge_len, y + PI_4 / edge_len
                cube_coords.append((panel, x_i, y_i))
                sphere_coords.append(convert_cube_to_sphere(panel, x_i, y_i))
    return cube_coords, sphere_coords


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


def calc_interpolated_feature(triangle_vertices, coeffs, all_coords, subject_features):
    # get features for each of the three closest points, add to a list in order of closest to farthest
    features = []
    for p in triangle_vertices:
        features_p = get_feature_for_point(p[0], p[1], all_coords, subject_features)
        features.append(features_p)

    # based on equation 6 in "3D Tune-In Toolkit: An open-source library for real-time binaural spatialisation"
    interpolated_feature = coeffs["alpha"] * features[0] + coeffs["beta"] * features[1] + coeffs["gamma"] * features[2]

    return interpolated_feature


def calc_all_interpolated_features(cs, features, feature_index, euclidean_sphere,
                                   euclidean_sphere_triangles, euclidean_sphere_coeffs):
    selected_feature_interpolated = []
    for i, p in enumerate(euclidean_sphere):
        if p[0] is not None:
            features_p = calc_interpolated_feature(triangle_vertices=euclidean_sphere_triangles[i],
                                                   coeffs=euclidean_sphere_coeffs[i],
                                                   all_coords=cs.get_all_coords(),
                                                   subject_features=features)
            selected_feature_interpolated.append(features_p[feature_index])
        else:
            selected_feature_interpolated.append(None)

    return selected_feature_interpolated
