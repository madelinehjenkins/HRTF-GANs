import numpy as np

PI_4 = np.pi / 4


def calc_offset(quadrant):
    return ((quadrant - 1) / 2) * np.pi


def calc_panel(elevation, azimuth):
    # use the azimuth to determine the quadrant of the sphere
    if azimuth < np.pi / 4:
        quadrant = 1
    elif azimuth < 3 * np.pi / 4:
        quadrant = 2
    elif azimuth < 5 * np.pi / 4:
        quadrant = 3
    else:
        quadrant = 4

    offset = calc_offset(quadrant)
    threshold_val = np.tan(elevation) / np.cos(azimuth - offset)
    # when close to the horizontal plane, must be panels 1 through 4 (inclusive)
    if -1 <= threshold_val < 1:
        return quadrant
    # above a certain elevation, in panel 5
    elif threshold_val >= 1:
        return 5
    # below a certain elevation, in panel 6
    elif threshold_val < -1:
        return 6


# used for obtaining cubed sphere coordinates from a pair of spherical coordinates
def convert_sphere_to_cube(elevation, azimuth):
    if elevation is None or azimuth is None:
        # if this position was not measured in the sphere, keep as np.nan in cube
        panel, x, y = np.nan, np.nan, np.nan
    else:
        # shift the range of azimuth angles such that it works with conversion equations
        if azimuth < -np.pi / 4:
            azimuth += 2 * np.pi
        panel = calc_panel(elevation, azimuth)

        if panel <= 4:
            offset = calc_offset(panel)
            x = azimuth - offset
            y = np.arctan(np.tan(elevation) / np.cos(azimuth - offset))
        elif panel == 5:
            x = np.arctan(np.sin(azimuth) / np.tan(elevation))
            y = np.arctan(-np.cos(azimuth) / np.tan(elevation))
        elif panel == 6:
            x = np.arctan(-np.sin(azimuth) / np.tan(elevation))
            y = np.arctan(-np.cos(azimuth) / np.tan(elevation))
    return panel, x, y


# used for obtaining spherical coordinates from cubed sphere coordinates
def convert_cube_to_sphere(panel, x, y):
    if panel <= 4:
        offset = calc_offset(panel)
        azimuth = x + offset
        elevation = np.arctan(np.tan(y) * np.cos(x))
    elif panel == 5:
        # if tan(x) is 0, handle as a special case
        if np.tan(x) == 0:
            azimuth = np.arctan(0)
            elevation = np.pi/2
        else:
            azimuth = np.arctan(-np.tan(x) / np.tan(y))
            elevation = np.arctan(np.sin(azimuth) / np.tan(x))
            if elevation < 0:
                elevation *= -1
                azimuth += np.pi
    # not including panel 6 for now, as it is being excluded from this data
    elif panel == 6:
        pass

    # ensure azimuth is in range -pi to +pi
    while azimuth > np.pi:
        azimuth -= 2 * np.pi
    while azimuth <= -np.pi:
        azimuth += 2 * np.pi

    return elevation, azimuth


def convert_sphere_to_cartesian(coordinates):
    x, y, z = [], [], []
    mask = []

    for elevation, azimuth in coordinates:
        if elevation is not None and azimuth is not None:
            mask.append(True)
            # convert to cartesian coordinates
            x_i = np.cos(elevation) * np.cos(azimuth)
            y_i = np.cos(elevation) * np.sin(azimuth)
            z_i = np.sin(elevation)

            x.append(x_i)
            y.append(y_i)
            z.append(z_i)
        else:
            mask.append(False)

    return np.asarray(x), np.asarray(y), np.asarray(z), mask


def convert_cube_to_cartesian(coordinates):
    x, y, z = [], [], []
    mask = []

    for panel, p, q in coordinates:
        if not np.isnan(p) and not np.isnan(q):
            mask.append(True)
            if panel == 1:
                x_i, y_i, z_i = PI_4, p, q
            elif panel == 2:
                x_i, y_i, z_i = -p, PI_4, q
            elif panel == 3:
                x_i, y_i, z_i = -PI_4, -p, q
            elif panel == 4:
                x_i, y_i, z_i = p, -PI_4, q
            elif panel == 5:
                x_i, y_i, z_i = -q, p, PI_4
            else:
                x_i, y_i, z_i = q, p, -PI_4

            x.append(x_i)
            y.append(y_i)
            z.append(z_i)
        else:
            mask.append(False)

    return np.asarray(x), np.asarray(y), np.asarray(z), mask
