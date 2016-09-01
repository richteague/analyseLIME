import numpy as np
from astropy.io import fits
import scipy.constants as sc
from astropy.io.fits import getval


class cube:

    def __init__(self, path, dist=None, inc=None, pa=None):

        # Data and axes.

        self.filename = path
        self.data = fits.getdata(self.filename, 0)
        self.linedata = np.array([self.data[i] - self.data[0]
                                  for i in range(self.data.shape[0])])
        self.velax = self.getVelocityAxis()
        self.posax = self.getPositionAxis()
        self.specax = self.getSpectralAxis()

        # Try and read in values from header.
        # Should work fine with makeLIME.py.

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
            except:
                print "No distance (.dist) given."
                self.dist = None
        else:
            self.dist = dist

        if pa is None:
            try:
                self.pa = getval(self.filename, 'pa', 0)
            except:
                print "No position angle (.pa) given."
                self.pa = None
        else:
            self.pa = pa

        # Commonly used values.

        self.x0 = getval(self.filename, 'crpix1', 0) - 1.
        self.y0 = getval(self.filename, 'crpix2', 0) - 1.
        self.nchan = self.velax.size
        self.npix = self.posax.size
        self.bunit = fits.getval(self.filename, 'bunit', 0)
        if self.dist is not None:
            self.projaxes = self.posax * self.dist
        self.x = self.posax[None, :]
        self.y = self.posax[:, None] / np.cos(self.inc)
        self.rvals = np.hypot(self.x, self.y)

        print 'Successfully read in %s.' % self.filename
        return

    # Some commonly used functions. #

    # Position axis in arcesconds.
    def getPositionAxis(self):
        a_len = getval(self.filename, 'naxis2', 0)
        a_del = getval(self.filename, 'cdelt2', 0)
        a_pix = getval(self.filename, 'crpix2', 0)
        return 3600. * ((np.arange(1, a_len+1) - a_pix) * a_del)

    # Velocity axis in meters per second.
    def getVelocityAxis(self):
        a_len = getval(self.filename, 'naxis3', 0)
        a_del = getval(self.filename, 'cdelt3', 0)
        a_pix = getval(self.filename, 'crpix3', 0)
        return (np.arange(1, a_len+1) - a_pix) * a_del

    # Get the spectral axis in Hertz.
    def getSpectralAxis(self):
        nu = getval(self.filename, 'restfreq', 0)
        return self.getVelocityAxis() * nu / sc.c

    # Clip the datacube to specific channels and remove continuum.
    def clipData(self, lowchan=0, highchan=-1, removeCont=1):
        data = self.data
        if removeCont:
            rgn = range(lowchan, self.nchan+1)[:highchan]
            data = np.array([data[i]-data[0] for i in rgn])
        else:
            data = data[lowchan:highchan]
        return np.where(data >= 0., data, 0.)

    # Clip the velocity axis and convert units if necessary.
    def clipVelo(self, lowchan, highchan, vunit='km/s'):
        velo = self.velax
        if 'kms' in vunit.replace('/', '').replace('per', ''):
            velo = np.array([velo[i] / 1e3
                             for i in range(lowchan, self.nchan+1)[:highchan]])
        else:
            velo = np.array([velo[i]
                             for i in range(lowchan, self.nchan+1)[:highchan]])
        return velo

    # Get the zeroth moment map.
    def getZeroth(self, lowchan=0, highchan=-1, removeCont=1,
                  bunit=None, vunit='km/s', mask=True):

        data = self.clipData(lowchan=lowchan, highchan=highchan,
                             removeCont=removeCont)
        data *= self.convertBunit(bunit)

        velo = self.clipVelo(lowchan=lowchan, highchan=highchan,
                             vunit=vunit)

        zeroth = np.trapz(data, dx=abs(np.diff(velo)[0]), axis=0)

        if mask:
            return zeroth * self.getNaNMask()
        else:
            return zeroth

    # Get the first moment map.
    # Method = 1: intensity weighted.
    # Method = 2: velocity of maximum emission.
    def getFirst(self, lowchan=0, highchan=-1, removeCont=1,
                 method=1, vunit='km/s', mask=True):

        data = self.clipData(lowchan=lowchan, highchan=highchan,
                             removeCont=removeCont)

        velo = self.clipVelo(lowchan=lowchan, highchan=highchan,
                             vunit=vunit)
        if method == 1:
            data = np.where(data == 0.0,
                            1e-30*np.random.random(data.shape),
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

    # Get the second moment map.
    def getSecond(self, lowchan=0, highchan=-1, removeCont=1,
                  vunit='km/s', mask=True):

        data = self.clipData(lowchan=lowchan, highchan=highchan,
                             removeCont=removeCont)
        velo = self.clipVelo(lowchan=lowchan, highchan=highchan,
                             vunit=vunit)

        first = self.getFirst(lowchan=lowchan, highchan=highchan,
                              method=1, vunit=vunit, mask=False)
        vcube = velo[:, None, None] * np.ones(data.shape)
        fcube = first[None, :, :] * np.ones(data.shape)
        data = np.where(data == 0.0, 1e-30*np.random.random(data.shape), data)
        second = np.average((vcube - fcube)**2, weights=data, axis=0)**0.5

        if mask:
            return second * self.getNaNMask()
        else:
            return second

    # Get the mask. Can be used for outlines.
    def getMask(self):
        if not hasattr(self, 'mask'):
            mask = np.sum(self.data, axis=0)
            self.mask = np.where(mask == mask.min(), 0, 1)
        return self.mask

    def getNaNMask(self):
        if not hasattr(self, 'nanmask'):
            mask = np.sum(self.data, axis=0)
            self.nanmask = np.where(mask == mask.min(), np.nan, 1)
        return self.nanmask

    # Get the radial profile of the zeroth moment.
    def getZerothProfile(self, bins=None, nbins=50, lowchan=0,
                         highchan=-1, removeCont=1, bunit=None,
                         vunit='km/s', mask=True):
        if bins is None:
            bins = np.linspace(0, 1.2*self.posax.max(), nbins+1)
        else:
            nbins = len(bins)-1

        ridxs = np.digitize(self.rvals, bins).ravel()
        zeroth = self.getZeroth(lowchan=lowchan, highchan=highchan,
                                removeCont=removeCont, bunit=bunit,
                                vunit=vunit, mask=mask).ravel()

        avg = np.array([np.nanmean(zeroth[ridxs == r])
                        for r in range(1, nbins+1)])
        std = np.array([np.nanstd(zeroth[ridxs == r])
                        for r in range(1, nbins+1)])
        rad = np.average([bins[1:], bins[:-1]], axis=0)
        return np.array([rad, avg, std])

    # Get the Radial profile of the second moment.
    def getSecondProfile(self, bins=None, nbins=50, lowchan=0,
                         highchan=-1, removeCont=1, vunit='km/s',
                         mask=True):
        if bins is None:
            bins = np.linspace(0, 1.2*self.posax.max(), nbins+1)
        else:
            nbins = len(bins)-1

        ridxs = np.digitize(self.rvals, bins).ravel()
        second = self.getSecond(lowchan=lowchan, highchan=highchan,
                                removeCont=removeCont, vunit=vunit,
                                mask=mask).ravel()

        avg = np.array([np.nanmean(second[ridxs == r])
                        for r in range(1, nbins+1)])
        std = np.array([np.nanstd(second[ridxs == r])
                        for r in range(1, nbins+1)])
        rad = np.average([bins[1:], bins[:-1]], axis=0)
        return np.array([rad, avg, std])

    # Convert brightness units.
    def convertBunit(self, bunit):
        if (self.bunit == bunit or bunit is None):
            print 'No change necessary.'
            scale = 1.
        elif 'mjy' in bunit.lower():
            print 'K to mJy/pix'
            scale = 1e3 / self.JanskytoKelvin()
        elif 'jy' in bunit.lower():
            print 'K to Jy/pix'
            scale = self.JanskytoKelvin()**-1
        elif 'k' in bunit.lower():
            print 'Jy/pix to K'
            scale = self.JanskytoKelvin()
        else:
            raise NotImplementedError("Unknown brightness unit.")
        return scale

    # Jansky to Kelvin conversion.
    def JanskytoKelvin(self):
        jy2k = 10**-26 * sc.c**2.
        jy2k /= 2. * fits.getval(self.filename, 'restfreq', 0)**2.
        jy2k /= sc.k * np.radians(fits.getval(self.filename, 'cdelt2', 0))**2.
        return jy2k

    # Plot the outline of the disk.
    def plotOutline(self, ax, level=0.99, lw=.5, c='k', ls='-'):
        ax.contour(self.posax, self.posax, self.getMask(), [level],
                   linewidths=[lw], colors=[c], linestyles=[ls])
        return

    def getIntegratedIntensity(self, removeCont=1, lowchan=0,
                               highchan=-1, bunit='jy', vunit='km/s'):
        data = self.clipData(lowchan=lowchan, highchan=highchan,
                             removeCont=removeCont)
        if bunit.lower() == 'k':
            data *= self.convertBunit('K')
        else:
            data *= self.convertBunit(bunit.split('/')[0]+'/pix')
        velo = self.clipVelo(lowchan=lowchan, highchan=highchan,
                             vunit=vunit)
        return velo, np.array([np.sum(c) for c in data])