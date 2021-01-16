"""
Utilities for loading measurement data from the UIUC aeronautical databases.
"""
import urllib.request as request
import urllib.parse as parse

import numpy as np
from scipy.interpolate import interp1d

UIUC_PROP_DB = "https://m-selig.ae.illinois.edu/props/"


def load_static_propeller(path,
                          urlopen_options=None,
                          interp_options=None):
    """Retrieve data about the static performance of a given propeller and
    provide interpolation functions for the thrust and power coefficient.

    Args:
      path: The relative path of the respective input file, e.g.
        `volume-2/data/apcff_4.2x4_static_0615rd.txt`
      urlopen_options: Dictionary of options passed to `urllib.request.urlopen`
        (Default value = None)
      interp_options: Dictionary of options passed to
        `scipy.interpolate.interp1d` (Default value = None)

    Returns:
      Functions for determining the thrust- and power-coefficient based on the
      speed (in 1/s)
    """

    urlopen_options = urlopen_options or {}
    interp_options = interp_options or {}

    full_url = parse.urljoin(UIUC_PROP_DB, path)

    with request.urlopen(full_url, **urlopen_options) as req:
        data = np.loadtxt(req, skiprows=1)
        speed = data[:, 0] / 60
        thrust_coeff = data[:, 1]
        power_coeff = data[:, 2]

        return interp1d(speed, thrust_coeff, **interp_options), \
            interp1d(speed, power_coeff, **interp_options)
