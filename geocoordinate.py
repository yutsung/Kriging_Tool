"""
"""

import numpy as np


def haversine(lon1d, lon2d, lat1d, lat2d):

    radius = earth_radius((lon1d + lon2d)/2)

    lon1 = lon1d * np.pi / 180.
    lon2 = lon2d * np.pi / 180.
    lat1 = lat1d * np.pi / 180.
    lat2 = lat2d * np.pi / 180.

    part1 = (np.sin( (lat1-lat2)/2) )**2
    part2 = np.cos(lat1) * np.cos(lat2) * (np.sin( (lon1-lon2)/2) )**2
    central_angle = 2 * np.arcsin(np.sqrt(part1 + part2))
    dis = radius * central_angle
    return dis


def earth_radius(lat):
    theta = lat/180 * np.pi
    r_equatorial = 6378.1370
    a = r_equatorial
    r_polar = 6356.7523
    b = r_polar

    r_theta = np.sqrt(
        ((a**2 * np.cos(theta))**2 + (b**2 * np.sin(theta))**2) /
        ((a    * np.cos(theta))**2 + (b    * np.sin(theta))**2)
    )
    return r_theta


