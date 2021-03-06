"""
Class to help read in the models from LIME. Basic class reutrns
the first three moments and the radial profile of the zeroth.
"""

from __future__ import print_function
import numpy as np
from astropy.io import fits
from astropy.io.fits import getval
import scipy.constants as sc


class cube:

    def __init__(self, path, **kwargs):

        self.filename = path

        # Read in the cube and the appropriate axes. Currently assume that the
        # position axes are the same.

        self.data = fits.getdata(self.filename, 0)
        self.velax = self.readVelocityAxis()
        self.nchan = self.velax.size
        self.posax = self.readPositionAxis()
        self.npix = self.posax.size
        self.specax = self.readSpectralAxis()
        self.unit = getval(self.filename, 'bunit', 0)

        # Parse the information about the geometry of the model from the
        # header. If not available, use the **kwargs to fill in. Should warn
        # if a value is not found and use np.nan as a filler.

        self.inc, self.pa, self.dist, self.azi = self.readHeader(**kwargs)

        # If the trio of (inc, pa, azi) are all defined, then we can calculate
        # the projected positions of each pixel assuming a thin disk.

        self.rvals, self.tvals = self.deprojectedCoords(**kwargs)

        # Remove the continuum for a quick analysis of the line emission.
        # Make a quick check that the first two and last two channels are the
        # same, otherwise raise a warning suggesting that continuum subtraction
        # may be slightly wrong.

        self.cont, self.line = self.subtractContinuum(**kwargs)
        self.total_intensity = self.getZeroth()

        return

    def deprojectedCoords(self, **kwargs):
        """Returns the deprojected (r, theta) coordinates."""

        # First check that the inclination and position angle are specified.

        if np.isnan(self.inc):
            print('No inclination specified. Assuming 0 rad.')
            self.inc = 0.0
        if np.isnan(self.pa):
            print('No position angle specified. Assuming 0 rad.')
            self.pa = 0.0

        # This is a two step process. First, derotate everything back to a
        # position angle of 0.0 (i.e. the major axis aligned with the x-axis),
        # then deproject along the y-axis.

        x_obs = self.posax[None, :] * np.ones(self.npix)[:, None]
        y_obs = self.posax[:, None] * np.ones(self.npix)[None, :]
        x_rot = x_obs * np.cos(self.pa) - y_obs * np.sin(self.pa)
        y_rot = x_obs * np.sin(self.pa) + y_obs * np.cos(self.pa)
        x_dep = x_rot
        y_dep = y_rot / np.cos(self.inc)
        return np.hypot(y_dep, x_dep), np.arctan2(y_dep, x_dep)

    def getZeroth(self, remove_continuum=True):
        """Returns the zeroth moment of the data."""
        if remove_continuum:
            return np.trapz(self.line, self.velax, axis=0)
        return np.trapz(self.data, self.velax, axis=0)

    def getFirst(self, method='weighted', **kwargs):
        """Returns the first moment of the data."""
        if not (method in 'weighted' or method in 'maximum'):
            raise ValueError("method must be 'weighted' or 'maximum'.")
        if method in 'weighted':
            return self.weightedFirst(**kwargs)
        return self.maximumFirst(**kwargs)

    def getSecond(self, **kwargs):
        """Returns the second moment of the data."""
        raise NotImplementedError()
        return

    def getIntensityProfile(self, bins=None, nbins=None, **kwargs):
        """Returns the azimutahlly averaged intensity profile."""
        bins = self.getRadialBins(bins=bins, nbins=nbins)
        nbins = bins.size - 1
        zeroth = self.getZeroth(**kwargs).ravel()
        ridxs = np.digitize(self.rvals.ravel(), bins)
        pvals = kwargs.get('percentiles', [0.16, 0.5, 0.94])
        rpnts = np.mean([bins[1:], bins[:-1]], axis=0)
        percentiles = [np.percentile(zeroth[ridxs == r], pvals)
                       for r in range(1, nbins+1)]
        percentiles = np.squeeze(percentiles).T
        print(percentiles.shape)

        # Convert from percentiles to median +/- error, better for plotting
        # fill_between style plots.

        if kwargs.get('uncertainty', True):
            profile = np.ones(percentiles.shape)
            profile[0] = percentiles[1]
            profile[1] = percentiles[1] - percentiles[0]
            profile[2] = percentiles[2] - percentiles[1]
            return rpnts, profile

        return rpnts, percentiles

    def getRadialBins(self, bins=None, nbins=None, **kwargs):
        """Returns the radial bins."""

        # bins specifies the bin edges used for np.digitize.
        # nbins, therefore, is len(bins) - 1.

        if bins is None and nbins is None:
            raise ValueError("Specify either 'bins' or 'nbins'.")
        if bins is not None and nbins is not None:
            raise ValueError("Specify either 'bins' or 'nbins'.")

        # By default return bins from 0 to the the edge of
        # the image.

        if bins is None:
            return np.linspace(0., self.posax.max(), nbins+1)
        return bins

    def weightedFirst(self, **kwargs):
        """First moment from intsensity weighted average."""
        emission = np.sum(self.line, axis=0)
        emission = np.where(emission != 0, 1, 0)
        emission = emission * np.ones(self.line.shape)
        noise = 1e-30 * np.random.random(self.line.shape)
        weights = np.where(emission, self.line, noise)
        vcube = self.velax[:, None, None] * np.ones(weights.shape)
        first = np.average(vcube, weights=weights, axis=0)
        emission = np.sum(emission, axis=0)
        return np.where(emission, first, kwargs.get('mask', np.nan))

    def maximumFirst(self, **kwargs):
        """First moment from the maximum value."""
        vidx = np.argmax(self.line, axis=0)
        vmax = np.take(self.velax, vidx)
        return np.where(vidx != 0, vmax, np.nan)

    def subtractContinuum(self, **kwargs):
        """Return the line-only data."""
        cchan = kwargs.get('cont_chan', 0)
        cont = self.data[cchan].copy()
        if not all([np.isclose(np.sum(cont - self.data[i]), 0)
                    for i in [1, -1, -2]]):
            print('Potential line emission in continuum channels.')
        line = np.array([chan - cont for chan in self.data])
        return cont, line

    def readVelocityAxis(self):
        """Return velocity axis in [km/s]."""
        a_len = getval(self.filename, 'naxis3', 0)
        a_del = getval(self.filename, 'cdelt3', 0)
        a_pix = getval(self.filename, 'crpix3', 0)
        return (np.arange(1, a_len+1) - a_pix) * a_del / 1e3

    def readPositionAxis(self):
        """Returns the position axis in ["]."""
        a_len = getval(self.filename, 'naxis2', 0)
        a_del = getval(self.filename, 'cdelt2', 0)
        a_pix = getval(self.filename, 'crpix2', 0)
        return 3600. * ((np.arange(1, a_len+1) - a_pix) * a_del)

    def readSpectralAxis(self):
        """Returns the spectral axis in Hz."""
        nu = getval(self.filename, 'restfreq', 0)
        return self.readVelocityAxis() * nu / sc.c

    def readHeader(self, **kwargs):
        """Reads the model properties."""
        inc = self.readValue('inc', **kwargs)
        pa = self.readValue('pa', **kwargs)
        dist = self.readValue('distance', **kwargs)
        azi = self.readValue('azi', **kwargs)
        return inc, pa, dist, azi

    def readValue(self, key, **kwargs):
        """Attempt to read a value from the header."""
        assert type(key) is str
        try:
            value = getval(self.filename, key, 0)
        except:
            value = kwargs.get(key, np.nan)
        if np.isnan(value):
            print('A value for', key, 'cannot be read.')
        return value
