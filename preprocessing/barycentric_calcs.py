import numpy as np


def calc_dist_haversine(elevation1, azimuth1, elevation2, azimuth2):
    # adapted from CalculateDistance_HaversineFomrula in the 3DTune-In toolkit
    # https://github.com/3DTune-In/3dti_AudioToolkit/blob/master/3dti_Toolkit/BinauralSpatializer/HRTF.cpp#L1052
    increment_azimuth = azimuth1 - azimuth2
    increment_elevation = elevation1 - elevation2
    sin2_inc_elev = np.sin(increment_elevation / 2) ** 2
    cos_elev1 = np.cos(elevation1)
    cos_elev2 = np.cos(elevation2)
    sin2_inc_azi = np.sin(increment_azimuth / 2) ** 2
    raiz = sin2_inc_elev + (cos_elev1 * cos_elev2 * sin2_inc_azi)
    sqrt_distance = raiz ** 0.5
    distance = 2 * np.arcsin(sqrt_distance)
    return distance


def calc_spherical_excess(elevation1, azimuth1, elevation2, azimuth2, elevation3, azimuth3):
    dist12 = calc_dist_haversine(elevation1, azimuth1, elevation2, azimuth2)
    dist13 = calc_dist_haversine(elevation1, azimuth1, elevation3, azimuth3)
    dist23 = calc_dist_haversine(elevation2, azimuth2, elevation3, azimuth3)
    semiperim = 0.5 * (dist12 + dist13 + dist23)
    inner = np.tan(0.5 * semiperim) * \
            np.tan(0.5 * (semiperim - dist12)) * \
            np.tan(0.5 * (semiperim - dist13)) * \
            np.tan(0.5 * (semiperim - dist23))
    excess = 4 * np.arctan(np.sqrt(inner))
    return excess


def calc_all_distances(elevation, azimuth, sphere_coords):
    distances = []
    for elev, azi in sphere_coords:
        if elev is not None and azi is not None:
            dist = calc_dist_haversine(elevation1=elevation, azimuth1=azimuth,
                                       elevation2=elev, azimuth2=azi)
            distances.append((elev, azi, dist))

    # sorted list of (elevation, azimuth, distance) for all points
    return sorted(distances, key=lambda x: x[2])


# get alpha, beta, and gamma coeffs for barycentric interpolation (modified for spherical triangle
def calc_barycentric_coordinates(elevation, azimuth, closest_points):
    # not zero indexing var names in order to match equations in 3D Tune-In Toolkit paper
    elev1, elev2, elev3 = closest_points[0][0], closest_points[1][0], closest_points[2][0]
    azi1, azi2, azi3 = closest_points[0][1], closest_points[1][1], closest_points[2][1]

    # modified calculations to suit spherical triangle
    denominator = calc_spherical_excess(elev1, azi1, elev2, azi2, elev3, azi3)

    alpha = calc_spherical_excess(elevation, azimuth, elev2, azi2, elev3, azi3) / denominator
    beta = calc_spherical_excess(elev1, azi1, elevation, azimuth, elev3, azi3) / denominator
    gamma = 1 - alpha - beta

    return {"alpha": alpha, "beta": beta, "gamma": gamma}
