import os
import numpy as np
from astropy.io import fits
import scipy.constants as sc
from astropy.io.fits import getval
from astropy.convolution import convolve
from matplotlib.patches import Ellipse

class beamclass:
    """Beam object to pass to convolution functions."""

    def __init__(self, beamparams):

        if type(beamparams) is float:
            self.maj = beamparams
            self.min = beamparams
            self.pa = 0.0
        elif len(beamparams) == 1:
            self.min = beamparams[0]
            self.maj = self.min
            self.pa = 0.0
        elif len(beamparams) == 3:
            self.min, self.maj, self.pa = beamparams
        else:
            raise ValueError("beamparams = [b_min, b_maj, b_pa]")

        if self.min > self.maj:
            tmp = self.min
            self.min = self.maj
            self.maj = tmp

        self.pa = np.radians(self.pa % 360.)
        self.area = np.pi * self.min * self.maj / 4. / np.log(2)

        return


class cubeclass:
    """Datacube read in from LIME."""

    def __init__(self, path, dist=None, inc=None, pa=None,
                 quick_convolve=True):
        """Read in the data and calculate the cube axes."""
        self.filename = path
        self.data = fits.getdata(self.filename, 0)
        self.velax = self.getVelocityAxis()
        self.posax = self.getPositionAxis()
        self.specax = self.getSpectralAxis()

        # Try and read in values from header, should work with makeLIME.py.
        if inc is None:
            try:
                self.inc = getval(self.filename, 'inc', 0)
            except:
                print "No inclination (.inc) given. Assuming face-on."
                self.inc = 0.0
        else:
            self.inc = inc

        if dist is None:
            try:
                self.dist = getval(self.filename, 'distance', 0)
                self.projaxes = self.posax * self.dist
            except:
                print "No distance (.dist) given."
                self.dist = None
        else:
            self.dist = dist
            self.projaxes = self.posax * self.dist

        if pa is None:
            try:
                self.pa = getval(self.filename, 'pa', 0)
            except:
                print "No position angle (.pa) given."
                self.pa = None
        else:
            self.pa = pa

        # Commonly used values.
        self.quick_convolve = quick_convolve
        self.x0 = getval(self.filename, 'crpix1', 0) - 1.
        self.y0 = getval(self.filename, 'crpix2', 0) - 1.
        self.nchan = self.velax.size
        self.npix = self.posax.size

        # Parse the brightness unit.
        # Should make it easier to calculate converions.
        self.bunit, self.tounit = self.brightnessconversions()

        self.x = self.posax[None, :]
        self.y = self.posax[:, None] / np.cos(self.inc)
        self.pixscale = np.diff(self.posax)[0]
        self.rvals = np.hypot(self.x, self.y)
        self.convolved_cubes = {}
        self.convolved_zeroths = {}
        return


    def pixtobeam(self, beam):
        """Jy/beam = Jy/pix * pixtobeam()"""
        return beam.area / np.diff(self.posax)[0]**2


    def beamKernel(self, beam):
        """Generate a beam kernel for the convolution."""
        sig_x = beam.maj / self.pixscale / 2. / np.sqrt(2. * np.log(2.))
        sig_y = beam.min / self.pixscale / 2. / np.sqrt(2. * np.log(2.))
        grid = np.arange(-np.round(sig_x) * 8, np.round(sig_x) * 8 + 1)
        a = np.cos(beam.pa)**2 / 2. / sig_x**2
        a += np.sin(beam.pa)**2 / 2. / sig_y**2
        b = np.sin(2. * beam.pa) / 4. / sig_y**2
        b -= np.sin(2. * beam.pa) / 4. / sig_x**2
        c = np.sin(beam.pa)**2 / 2. / sig_x**2
        c += np.cos(beam.pa)**2 / 2. / sig_y**2
        kernel = c * grid[:, None]**2 + a * grid[None, :]**2
        kernel += 2 * b * grid[:, None] * grid[None, :]
        return np.exp(-kernel) / 2. / np.pi / sig_x / sig_y


    def convolveChannel(self, channel, beam):
        """Returns the convolved channel or moment map with the beam."""
        return convolve(channel, self.beamKernel(beam))

        self.convolved_zeroths[beam_maj, beam_min, beam_pa] = conv_zeroth
        return self.convolved_zeroths[beam_maj, beam_min, beam_pa]


    def convolveCube(self, beam):
        """Convolve the datacube with a beam."""
        try:
            return self.convolved_cubes[beam_maj, beam_min, beam_pa]
        except:
            print 'Convolving cube. May take a while.'
        cube = [self.convolveChannel(chan, beam) for chan in self.data]
        self.convolved_cubes[beam_maj, beam_min, beam_pa] = np.array(cube)
        return self.convolved_cubes[beam_maj, beam_min, beam_pa]


    def getPositionAxis(self):
        """Returns the position axis in arcseconds."""
        a_len = getval(self.filename, 'naxis2', 0)
        a_del = getval(self.filename, 'cdelt2', 0)
        a_pix = getval(self.filename, 'crpix2', 0)
        return 3600. * ((np.arange(1, a_len+1) - a_pix) * a_del)


    def getVelocityAxis(self):
        """Returns the velocity axis in m/s."""
        a_len = getval(self.filename, 'naxis3', 0)
        a_del = getval(self.filename, 'cdelt3', 0)
        a_pix = getval(self.filename, 'crpix3', 0)
        return (np.arange(1, a_len+1) - a_pix) * a_del


    def getSpectralAxis(self):
        """Returns the spectral axis in Hz."""
        nu = getval(self.filename, 'restfreq', 0)
        return self.getVelocityAxis() * nu / sc.c


    def clipData(self, low=0, high=-1, removeCont=1, beam=None):
        """Clip the datacube to specific channels and remove continuum. If a
        list of beam parameters are specified, use the convolved cube instead.
        Continuum subtraction just models the first channel as continuum."""

        if beam is None or self.quick_convolve:
            data = self.data.copy()
        else:
            data = self.convolveCube(beam)

        if removeCont:
            rgn = range(low, self.nchan+1)[:high]
            data = np.array([data[i]-data[0] for i in rgn])
        else:
            data = data[low:high]

        return np.where(data >= 0., data, 0.)


    def clipVelo(self, low, high, vunit='km/s'):
        """Clip the velocity axis and convert units if necessary."""
        velo = self.velax.copy()
        velo = [velo[i] for i in range(low, self.nchan+1)[:high]]
        velo = np.array(velo)
        if 'kms' in vunit.replace('/', '').replace('per', ''):
            velo /= 1e3
        return np.array(velo)


    def getZeroth(self, low=0, high=-1, removeCont=1, bunit=None,
                  vunit='km/s', mask=True, beamparams=None):
        """Returns the zeroth moment map. Convolve if necessary."""
        if beamparams is not None:
            beam = beamclass(beamparams)
        else:
            beam = None
        # Calculate a quick zeroth momenet. TODO: Include a dictionary here.
        velo = self.clipVelo(low=low, high=high, vunit=vunit)
        data = self.clipData(low=low, high=high, removeCont=removeCont, beam=beam)
        data *= self.convertBunit(bunit, beam=beam)
        zeroth = np.trapz(data, dx=abs(np.diff(velo)[0]), axis=0)
        # If quick_convolve, only convolve the zeroth moment, else the whole cube.
        if (beam is not None and self.quick_convolve):
            try:
                zeroth = self.convolved_zeroths[beam.maj, beam.min, beam.pa, bunit.lower()]
            except:
                zeroth = self.convolveChannel(zeroth, beam)
                self.convolved_zeroths[beam.maj, beam.min, beam.pa, bunit.lower()] = zeroth
        # Return a masked, or unmasked, zeroth moment.
        if mask:
            return zeroth * self.getMask()
        else:
            return zeroth

    def getZerothProfile(self, bins=None, nbins=50, low=0, high=-1,
                         removeCont=1, bunit=None, vunit='km/s', mask=True,
                         beamparams=None):
        """Get the radial profile of the zeroth moment."""
        # Generate the radial grid if not specified.
        if bins is None:
            bins = np.linspace(0, 1.2*self.posax.max(), nbins+1)
        else:
            nbins = len(bins)-1
        ridxs = np.digitize(self.rvals.ravel(), bins)
        # Return the zeroth moment with requested parameters.
        zeroth = self.getZeroth(low=low, high=high, removeCont=removeCont,
                                bunit=bunit, vunit=vunit, mask=mask,
                                beamparams=beamparams).ravel()
        # Calculate the mean and standard devation of each radial bin.
        avg = [np.nanmean(zeroth[ridxs == r]) for r in range(1, nbins+1)]
        std = [np.nanstd(zeroth[ridxs == r]) for r in range(1, nbins+1)]
        rad = np.average([bins[1:], bins[:-1]], axis=0)
        return np.array([rad, avg, std])


    def getFirst(self, low=0, high=-1, removeCont=1, method=1, vunit='km/s',
                 mask=True, beam=None):
        """Calculates the first moment by two methods: 1, intensity weighted;
        2, velocity of maximum emission."""

        velo = self.clipVelo(low=low, high=high, vunit=vunit)
        data = self.clipData(low=low, high=high, removeCont=removeCont,
                             beam=beam)
        if method == 1:
            data = np.where(data == 0.0,
                            1e-30 * np.random.random(data.shape),
                            data)
            first = np.average(velo[:, None, None] * np.ones(data.shape),
                               weights=data, axis=0)
        elif method == 2:
            first = np.array([[velo[data[:, j, i].argmax()]
                               for i in range(self.npix)]
                              for j in range(self.npix)])
        else:
            raise ValueError("Method must be 1 or 2. See help for more.")
        if mask:
            return first * self.getMask()
        else:
            return first


    def getSecond(self, low=0, high=-1, removeCont=1,
                  vunit='km/s', mask=True, beam=None):
        """Returns the second moment map."""

        velo = self.clipVelo(low=low, high=high, vunit=vunit)
        data = self.clipData(low=low, high=high, removeCont=removeCont,
                             beam=beam)
        first = self.getFirst(low=low, high=high, method=1, vunit=vunit,
                              mask=False)
        vcube = velo[:, None, None] * np.ones(data.shape)
        fcube = first[None, :, :] * np.ones(data.shape)
        data = np.where(data == 0.0, 1e-30*np.random.random(data.shape), data)
        second = np.average((vcube - fcube)**2, weights=data, axis=0)**0.5
        if mask:
            return second * self.getNaNMask()
        else:
            return second

    def getMask(self, maskval=np.nan):
        """Mask non-disk model points."""
        if not hasattr(self, 'mask'):
            mask = np.sum(self.data, axis=0)
            self.mask = np.where(mask == mask.min(), maskval, 1)
        return self.mask

    def getNaNMask(self):
        raise NotImplementedError("Change!")
        if not hasattr(self, 'nanmask'):
            mask = np.sum(self.data, axis=0)
            self.nanmask = np.where(mask == mask.min(), np.nan, 1)
        return self.nanmask



    # Get the Radial profile of the second moment.
    def getSecondProfile(self, bins=None, nbins=50, lowchan=0,
                         highchan=-1, removeCont=1, vunit='km/s',
                         mask=True, beam=None):
        if bins is None:
            bins = np.linspace(0, 1.2*self.posax.max(), nbins+1)
        else:
            nbins = len(bins)-1

        ridxs = np.digitize(self.rvals, bins).ravel()
        second = self.getSecond(lowchan=lowchan, highchan=highchan,
                                removeCont=removeCont, vunit=vunit,
                                mask=mask, beam=beam).ravel()

        avg = np.array([np.nanmean(second[ridxs == r])
                        for r in range(1, nbins+1)])
        std = np.array([np.nanstd(second[ridxs == r])
                        for r in range(1, nbins+1)])
        rad = np.average([bins[1:], bins[:-1]], axis=0)
        return np.array([rad, avg, std])

    def brightnessconversions(self):
        """Calculate the conversion brightness unit factors."""
        # Brightnes unit of file.
        bunit = fits.getval(self.filename, 'bunit', 0).lower()
        bunit = bunit.replace('/', '').replace('per', '').replace(' ', '')
        if bunit not in ['jypixel', 'k']:
            raise ValueError('Cannot read brightness unit.')
        # Conversion dictionary.
        tounit = {}
        if bunit == 'jypixel':
            tounit['jypixel'] = 1.
            tounit['mjypixel'] = 1e3
            tounit['k'] = self.JanskytoKelvin()
        else:
            tounit['k'] = 1.
            tounit['jypixel'] = 1. / self.JanskytoKelvin()
            tounit['mjypixel'] = 1e-3 / self.JanskytoKelvin()
        return bunit, tounit

    def convertBunit(self, bunit, beam=None):
        """Convert the brightness units to bunit."""
        if bunit is None:
            return 1.
        bunit = bunit.lower().replace('/', '')
        bunit = bunit.replace('per', '').replace(' ', '')
        if bunit not in ['jybeam', 'mjybeam', 'jypixel', 'mjypixel', 'k']:
            raise ValueError("bunit must be (m)Jy/pixel, (m)Jy/beam or K.")
        # Convert to requested unit.
        if bunit == 'k':
            return self.tounit['k']
        elif 'mjy' in bunit:
            rescale = self.tounit['mjypixel']
        elif 'jy' in bunit:
            rescale = self.tounit['jypixel']
        # Add the beam rescaling.
        if ('beam' in bunit and beam is not None):
            rescale *= self.pixtobeam(beam)
        elif ('beam' in bunit and beam is None):
            print 'Warning, no beam specified! Reverting to per pixel.'
        return rescale


    def JanskytoKelvin(self, beam=None):
        """Jansky to Kelvin conversion. If no beam specified will return Jy/pix,
        else will return Jy/beam."""
        jy2k = 10**-26 * sc.c**2.
        jy2k /= 2. * fits.getval(self.filename, 'restfreq', 0)**2.
        jy2k /= sc.k
        if beam is None:
            jy2k /= np.radians(fits.getval(self.filename, 'cdelt2', 0))**2.
        else:
            jy2k /= beam
        return jy2k


    def getIntegratedIntensity(self, removeCont=1, low=0, high=-1, bunit='Jy', vunit='km/s'):
        """Return the integrated intensity."""
        data = self.clipData(low=low, high=high, removeCont=removeCont)
        velo = self.clipVelo(low=low, high=high, vunit=vunit)
        if bunit.lower() == 'k':
            bunit = 'k'
        elif 'mjy' in bunit.lower():
            bunit = 'mjyperpixel'
        elif 'jy' in bunit.lower():
            bunit = 'jyperpixel'
        else:
            raise ValueError('bunit must be either mJy, Jy or K.')
        data *= self.convertBunit(bunit)
        return velo, np.array([np.sum(c) for c in data])


    def plotBeam(self, ax, beamparams, x=0.1, y=0.1, color='k'):
        """Plot the beam on the supplied axes."""
        if beamparams is None:
            return
        beam = beamclass(beamparams)
        x_pos = x * (ax.get_xlim()[1] - ax.get_xlim()[0]) + ax.get_xlim()[0]
        y_pos = y * (ax.get_ylim()[1] - ax.get_ylim()[0]) + ax.get_ylim()[0]
        ax.add_patch(Ellipse((x_pos, y_pos),
                             width = beam.min,
                             height = beam.maj,
                             angle = np.degrees(beam.pa),
                             fill = False,
                             hatch = '////',
                             lw = 1.25,
                             color = color,
                             transform = ax.transData))
        return


    def plotOutline(self, ax, level=0.99, lw=.5, c='k', ls='-'):
        """Plot the outline of the emission on teh supplied axes."""
        ax.contour(self.posax, self.posax, self.getMask(), [level],
                   linewidths=[lw], colors=[c], linestyles=[ls])
        return
