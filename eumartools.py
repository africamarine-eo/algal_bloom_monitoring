"""
copyright: 2022 EUMETSAT
license: ../LICENSE.txt

A general tools module with functions for manipulating EUMETSAT marine data.

This module contains functions for manipulating EUMETSAT marine data. It is part
of the eumartools toolkit.

Edited by Marie Smith (CSIR) from https://gitlab.com/benloveday/eumartools.git
"""
import xarray as xr
import numpy as np
from skimage import exposure
from matplotlib.path import Path


def flag_mask(flag_file, flag_variable, applied_flags, test=False, dtype=None):
    """Function to build a binary mask from specific flags.

    Args:
        flag_file (string): the netCDF format file containing the flags
        flag_variable (string): the name of the flag field in flag_file
        applied_flags (list): names of flags to apply, as list of strings
        test (bool): test case switch

    Returns:
        if succesful & test is true (int); sum of flag binary flag mask
        if succesful & test is false (numpy.ndarray); binary flag mask
        if unsuccessful (string); error message

    """
    try:
        flag_fid = xr.open_dataset(flag_file)
        flags = flag_fid.get(flag_variable).data
        flag_names = flag_fid.get(flag_variable).flag_meanings.split(" ")
        flag_values = flag_fid.get(flag_variable).flag_masks
        flag_fid.close()
    except OSError as error:
        msg = "Unsuccessful!", error, "occurred."
        print(msg)
        return msg

    try:
        if not dtype:
            flag_bits = np.zeros(np.shape(flags), np.dtype(type(flags[0][0])))
        else:
            flag_bits = np.zeros(np.shape(flags), dtype)

        for flag in applied_flags:
            try:
                flag_bits = flag_bits | flag_values[flag_names.index(flag)]
            except TypeError:
                print(flag + " not present")

        if test:
            return np.sum(np.sum((flags & flag_bits) > 0))

        return (flags & flag_bits) > 0

    except Exception as error:
        msg = "Unsuccessful!", error, "occurred."
        print(msg)
        return msg


def point_distance(lon1, lon2, lat1, lat2, mode="global"):
    """Function to calculate the distance between a lat/lon point and lat/lon
       arrays.

    Args:
        lon1 (float): longitude coordinate
        lon2 (numpy array): longitude array
        lat1 (float): latitude coordinate
        lat2 (numpy array): latitude array

    Returns:
        if successful, array of distances
        else returns an error

    """

    try:
        R_earth = 6367442.76

        if mode == "global":
            # Compute the distances across the globe
            phi1 = (90 - lat1) * np.pi / 180
            phi2 = (90 - lat2) * np.pi / 180
            theta1 = lon1 * np.pi / 180
            theta2 = lon2 * np.pi / 180
            cos = np.sin(phi1) * np.sin(phi2) * np.cos(
                theta1 - theta2) + np.cos(phi1) * np.cos(phi2)
            dist = R_earth * np.arccos(cos)
        elif mode == "local":
            # Compute the distances with local approximation (no curve)
            dist = R_earth * ( (lon2 - lon1)**2 + (lat2 - lat1)**2 )**0.5
        return dist
    except Exception as error:
        msg = "Unsuccessful!", error, "occurred."
        print(msg)
        return msg


def subset_image(grid_lon, grid_lat, lons, lats, mode="global"):
    """Function to cut a box out of an image using the grid indices
        for the image corners. BEWARE USING THIS ON HALF-ORBIT,
        FULL-ORBIT or POLAR DATA.

    Args:
        grid_lon (numpy array): longitude array
        grid_lat (numpy array): latitude array
        lons (list): list of vertex longitudes
        lats (list): list of vertex latitudes
        mode (string): 'global' or 'local'

    Returns:
        if successful, returns ij indexes of box corners.
        else returns an error

    """

    try:
        nx = np.shape(grid_lon)[0]
        ny = np.shape(grid_lon)[1]
        poly_verts = []
        extracted_x = []
        extracted_y = []
        for lon,lat in zip(lons, lats):
            dist = point_distance(lon, grid_lon, lat,
                          grid_lat, mode=mode)
            i0, j0 = np.unravel_index(dist.argmin(), dist.shape)
            poly_verts.append((i0, j0))
            extracted_x.append(j0)
            extracted_y.append(i0)

        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T

        path = Path(poly_verts)
        mask = path.contains_points(points)
        mask = mask.reshape((ny,nx)).astype(float)
        mask[mask == 0.0]=np.nan

        return extracted_x, extracted_y, mask
    except Exception as error:
        msg = "Unsuccessful!", error, "occurred."
        print(msg)
        return msg
