import os
import numpy as np
from string import letters
from astropy.io import fits
import scipy.constants as sc
from astropy.io.fits import getval
from astropy.convolution import convolve
from matplotlib.patches import Ellipse
from scipy.ndimage.interpolation import rotate

class beamclass:
    """Beam object to pass to convolution functions."""
    def __init__(self, beamparams):
        if beamparams is None:
            self.maj = None
            self.min = None
            self.pa = None
            return
        elif type(beamparams) is float:
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
        self.eff = np.sqrt(self.min * self.maj)
        return


class cubeclass:
    """Datacube read in from LIME."""
    def __init__(self, path, dist=None, inc=None, pa=None):
        self.filename = path
        self.velax = self.getVelocityAxis()
        self.posax = self.getPositionAxis()
        self.specax = self.getSpectralAxis()
        self.data = fits.getdata(self.filename, 0)
        self.cont = self.data[0][None, :, :] * np.ones(self.velax.size)[:, None, None]
        self.line = None

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
        self.x0 = getval(self.filename, 'crpix1', 0) - 1.
        self.y0 = getval(self.filename, 'crpix2', 0) - 1.
        self.nchan = self.velax.size
        self.npix = self.posax.size

        self.bunit, self.tounit = self.brightnessconversions()
        self.x = self.posax[None, :]
        self.y = self.posax[:, None] / np.cos(self.inc)
        self.pixscale = abs(getval(self.filename, 'cdelt1', 0)) * 3600.
        self.rvals = np.hypot(self.x, self.y)
        self.tvals = np.arctan2(self.y, self.x)
        self.conv_zeroths = {}
        self.conv_chans = {}
        self.sgnlperchan = None
        self.line = None
        return


    def estimateSignal(self):
        """Estimates the signal per channel."""
        if self.sgnlperchan is None:
            linemax = np.amax(np.amax(self.onlyline(), axis=1), axis=1)
            self.sgnlperchan = np.percentile(linemax, 64.)
        return self.sgnlperchan

    def onlyline(self):
        """Returns the line emission."""
        if self.line is None:
            self.line = np.array([chan - self.data[0] for chan in self.data])
        return self.line

    def pixtobeam(self, beam):
        """Jy/beam = Jy/pix * pixtobeam()"""
        return beam.area / np.diff(self.posax)[0]**2

    def getPositionAxis(self):
        """Returns the position axis in arcseconds."""
        a_len = getval(self.filename, 'naxis2', 0)
        a_del = getval(self.filename, 'cdelt2', 0)
        a_pix = getval(self.filename, 'crpix2', 0)
        return 3600. * ((np.arange(1, a_len+1) - a_pix) * a_del)

    def getVelocityAxis(self):
        """Returns the velocity axis in km/s."""
        a_len = getval(self.filename, 'naxis3', 0)
        a_del = getval(self.filename, 'cdelt3', 0)
        a_pix = getval(self.filename, 'crpix3', 0)
        return (np.arange(1, a_len+1) - a_pix) * a_del / 1e3

    def getSpectralAxis(self):
        """Returns the spectral axis in Hz."""
        nu = getval(self.filename, 'restfreq', 0)
        return self.getVelocityAxis() * nu / sc.c
        
    def convolve2D(self, arr, **kwargs):
        """Convolves a 2D array with a beam."""
        beam = kwargs.get('beam', None)
        if beam is None:
            return arr
        return convolve(arr, self.beamKernel(beamclass(beam)))
    
    def noiseFactor(self, **kwargs):
        """Rescale the noise to account for convolution."""
        beam = kwargs.get('beam', None)
        if beam is None:
            return 1.
        beam = beamclass(beam) 
        return max(1, 1.07 * np.power(beam.eff / self.pixscale, 1.326))
        
    def addNoise(self, arr, onlychannel=False, **kwargs):
        """Add noise to the array (e.g. a channel or moment map)."""
        snr = kwargs.get('snr', None)
        if snr is None:
            return arr    
        if onlychannel:
            signal = self.estimateSignal()
        else:
            signal = np.nanmax(self.getZeroth())
        noise = signal * self.noiseFactor(**kwargs) / snr
        noise *= np.random.random(arr.shape)
        return arr + noise                
    
    def ZerothDict(self, **kwargs):
        """Keywords for the pre-calculated zeroth moment dictionary."""
        beam = beamclass(kwargs.get('beam', None))
        return (kwargs.get('removeCont', True), beam.min, beam.maj, 
                beam.pa, kwargs.get('snr', None))
        
    def getZeroth(self, **kwargs):
        """Returns the zeroth moment map."""
        
        # See if the moment has been previously calculated.       
        try:
            zeroth = self.conv_zeroths[self.ZerothDict(**kwargs)].copy()
            zeroth *= self.getMask(**kwargs) * self.convertBunit(**kwargs) 
            return zeroth
        except:
            pass
            
        # If not, calculate it.
        zeroth = np.trapz(self.data.copy(), self.velax, axis=0)
        zeroth = self.addNoise(zeroth, **kwargs)
        zeroth = self.convolve2D(zeroth, **kwargs)
        
        # Remove the continuum.
        if kwargs.get('removeCont', False):
            continuum = np.trapz(self.cont.copy(), self.velax, axis=0)
            continuum = self.addNoise(continuum, **kwargs)
            continuum = self.convolve2D(continuum, **kwargs)
            zeroth -= continuum
            
        # Add the dictionary of pre-calculated zeroth moments. 
        self.conv_zeroths[self.ZerothDict(**kwargs)] = zeroth.copy()
        zeroth *= self.getMask(**kwargs) * self.convertBunit(**kwargs) 
        return zeroth

    def getZerothProfile(self, bins=None, nbins=50, **kwargs):
        """Returns the radial profile of the zeroth moment."""
        
        # Determine the bins.
        if bins is None:
            bins = np.linspace(0.05, 1.2 * self.posax.max(), nbins + 1)
        else:
            nbins = len(bins) - 1
        ridxs = np.digitize(self.rvals.ravel(), bins)
        
        # Calculate the average and standard deviation for each bin.
        zeroth = np.nan_to_num(self.getZeroth(**kwargs).ravel())
        avg = [np.nanmean(zeroth[ridxs == r]) for r in range(1, nbins+1)]
        std = [np.nanstd(zeroth[ridxs == r]) for r in range(1, nbins+1)]
        rad = np.average([bins[1:], bins[:-1]], axis=0)
        return np.array([rad, avg, std])



    def getMaximum(self, removeCont=True, bunit=None, beamparams=None):
        raise NotImplementedError('Not done yet!')
        return

    def chanidx(self, c):
        """Return the channel index. If c is a string, assume a velocity."""
        if type(c) is str:
            vel = float(c.translate(None, letters).replace('/', ''))
            if not ('km' in c and 's' in c):
                vel /= 1e3
            c = abs(self.velax - vel).argmin()
        else:
            c = int(c)
        return c
        
    def ChannelDict(self, c, **kwargs):
        """Keywords for the pre-calculated channel dictionary."""
        beam = beamclass(kwargs.get('beam', None))
        return (self.chanidx(c), kwargs.get('removeCont', True), 
                beam.min, beam.maj, beam.pa, kwargs.get('snr', None))


    def getChannel(self, c, **kwargs):
        """Returns a channel. Specify either a channel index for velocity."""
        
        # First see if the channel has already been called.
        try:
            chan = self.conv_chans[self.ChannelDict(c, **kwargs)].copy()
            return chan * self.getMask(**kwargs) * self.convertBunit(**kwargs) 
        except:
            pass
            
        # If not, calculate it.
        chan = self.data[self.chanidx(c)].copy()
        chan = self.addNoise(chan, onlychannel=True, **kwargs)
        chan = self.convolve2D(chan, **kwargs)
        
        # Remove the continuum.           
        if kwargs.get('removeCont', True):
            continuum = self.data[0].copy()
            continuum = self.addNoise(continuum, onlychannel=True, **kwargs)
            continuum = self.convolve2D(continuum, **kwargs)
            chan -= continuum
        
        # Add to the dictionary of pre-calculated channels and return.
        self.conv_chans[self.ChannelDict(c, **kwargs)] = chan.copy()
        return chan * self.getMask(**kwargs) * self.convertBunit(**kwargs)


    def getMaximumProfile(self, bins=None, nbins=50, removeCont=True):
        """Return the maximum brightness along the line of sight."""
        if bins is None:
            bins = np.linspace(0, 1.2 * self.posax.max(), nbins + 1)
        else:
            nbins = len(bins) - 1
        ridxs = np.digitize(self.rvals.ravel(), bins)
        if removeCont:
            if self.line is None:
                self.line = [chan - self.data[0] for chan in self.data]
                self.line = np.array(self.line)
            maxvals = np.amax(self.line, axis=0)
        else:
            maxvals = np.amax(self.data, axis=0)
        maxvals *= self.convertBunit(bunit='K')
        maxvals = maxvals.ravel()
        avg = [np.nanmean(maxvals[ridxs == r]) for r in range(1, nbins+1)]
        std = [np.nanstd(maxvals[ridxs == r]) for r in range(1, nbins+1)]
        rad = np.average([bins[1:], bins[:-1]], axis=0)
        return np.array([rad, avg, std])


    def getFirst(self, **kwargs):
        """Returns the first moment of the data."""
        weights = self.onlyline()
        noemission = np.where(np.sum(weights, axis=0) == 0, True, False)
        weights = np.where(weights == 0.0, 
                           1e-30 * np.random.random(weights.shape),
                           weights)
        vcube = self.velax[:, None, None] * np.ones(weights.shape)
        first = np.average(vcube, weights=weights, axis=0)
        first = np.where(noemission, 0., first)
        first = self.convolve2D(first, **kwargs)
        return first * self.getMask(**kwargs)

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

    def getMask(self, **kwargs):
        """Returns a mask for moment maps."""
        if kwargs.get('mask', False):
            mask = np.sum(self.data, axis=0)
            fill = kwargs.get('maskval', np.nan)
            return np.where(mask == mask.min(), fill, 1.)
        else:
            return 1.
            
    def getFluxDensity(self, bunit='Jy', removeCont=True):
        """Return the flux density."""
        if removeCont:
            tosum = self.onlyline().copy()
        else:
            tosum = self.data.copy()
        if bunit not in ['Jy', 'mJy']:
            raise ValueError('bunit must be (m)Jy.')
        tosum *= self.convertBunit(bunit=bunit+'/pixel')
        return self.velax, np.array([np.sum(c) for c in tosum])

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


    #### Convolution functions. ################################################

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



    #### Brightness unit conversions. ##########################################

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

    def parseBunitKwargs(self, **kwargs):
        """Parse the kwargs appropriate for the brightness unit conversion."""
        try:
            bunit = kwargs['bunit']
        except:
            bunit = None
        if bunit is not None:
            bunit = bunit.lower().replace('/', '')
            bunit = bunit.replace('per', '').replace(' ', '')
            if bunit not in ['jybeam', 'mjybeam', 'jypixel', 'mjypixel', 'k']:
                raise ValueError("bunit must be (m)Jy/pixel, (m)Jy/beam or K.")
        try:
            beam = kwargs['beam']
            if beam is not None:
                beam = beamclass(beam)
        except:
            beam = None
        return bunit, beam

    def convertBunit(self, **kwargs):
        """Convert the brightness units to bunit."""
        bunit, beam = self.parseBunitKwargs(**kwargs)  
        if bunit is None:
            return 1.
        elif bunit == 'k':
            return self.tounit['k']
        elif 'mjy' in bunit:
            rescale = self.tounit['mjypixel']
        elif 'jy' in bunit:
            rescale = self.tounit['jypixel']
            
            
        if ('beam' in bunit and beam is not None):
            rescale *= self.pixtobeam(beam)
        elif ('beam' in bunit and beam is None):
            print 'Warning, no beam specified! Reverting to per pixel.'
        return rescale

    def JanskytoKelvin(self):
        """Jansky to Kelvin conversion."""
        jy2k = 10**-26 * sc.c**2. / sc.k
        jy2k /= 2. * fits.getval(self.filename, 'restfreq', 0)**2.
        jy2k /= np.radians(fits.getval(self.filename, 'cdelt2', 0))**2.
        return jy2k


    #### Plotting helper functions. ############################################

    def plotIsoRadius(self, isoradius, M_star, vunit='km/s'):
        """Plot contours of iso-radius in the PV diagram."""
        p = self.posax * self.dist * sc.au
        r = isoradius * sc.au
        v = np.sqrt(sc.G * M_star * 2e30 / r) * np.sin(self.inc)
        return v * np.cos(np.arctan2(np.sqrt(r**2 - p**2), p))



    def plotBeam(self, ax, beamparams, x=0.1, y=0.1, color='k',
                 hatch='/////////', linewidth=1.25):
        if beamparams is None:
            return
        beam = beamclass(beamparams)
        x_pos = x * (ax.get_xlim()[1] - ax.get_xlim()[0]) + ax.get_xlim()[0]
        y_pos = y * (ax.get_ylim()[1] - ax.get_ylim()[0]) + ax.get_ylim()[0]
        ax.add_patch(Ellipse((x_pos, y_pos), width=beam.min, height=beam.maj,
                             angle=np.degrees(beam.pa), fill=False, hatch=hatch,
                             lw=linewidth, color=color, transform=ax.transData))
        return


    def plotOutline(self, ax, level=0.99, lw=.5, c='k', ls='-'):
        """Plot the outline of the emission."""
        ax.contour(self.posax, self.posax, self.getMask(), [level],
                   linewidths=[lw], colors=[c], linestyles=[ls])
        return
