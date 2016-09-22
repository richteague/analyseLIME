import numpy as np
from astropy.io import fits
import scipy.constants as sc
from astropy.io.fits import getval
from astropy.convolution import convolve
from astropy.modeling.models import Gaussian2D

class cube:

    def __init__(self, path, dist=None, inc=None, pa=None, 
                 quick_convolve=True):
        """Upon initialisation read in the data and calculate the cube axes.
        Will try and read the distance, inclination and position angle of the
        source from the header file."""
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
        self.bunit = fits.getval(self.filename, 'bunit', 0).lower()

        self.x = self.posax[None, :]
        self.y = self.posax[:, None] / np.cos(self.inc)
        self.pixscale = np.diff(self.posax)[0]
        self.rvals = np.hypot(self.x, self.y)
        self.convolved_cubes = {}
        self.convolved_zeroths = {}
        print 'Successfully read in %s.' % self.filename       
        return
    
    def pixtobeam(self, beam_maj, beam_min=None):
        """Jy/beam = Jy/pix * pixtobeam()"""
        pixarea = np.diff(self.posax)[0]**2
        if beam_min is None:
            beam_min = beam_maj
        beamarea = np.pi * beam_maj * beam_min / 4. / np.log(2)
        return beamarea / pixarea

    def beamKernel(self, beam_x, beam_y, beam_pa):
        """Generate a beam kernel for the convolution."""
         
        sig_x = beam_x / self.pixscale / 2. / np.sqrt(2. * np.log(2.))
        if beam_y is None:
            sig_y = sig_x
        else:

            sig_y = beam_y / self.pixscale / 2. / np.sqrt(2. * np.log(2.))
        theta = np.radians(beam_pa % 360.)
        grid = np.arange(-np.round(sig_x) * 8, np.round(sig_x) * 8 + 1)
        
        # Calculate and return the kernel. 
        # https://en.wikipedia.org/wiki/Gaussian_function
        a = np.cos(theta)**2 / 2. / sig_x**2
        a += np.sin(theta)**2 / 2. / sig_y**2
        b = np.sin(2. * theta) / 4. / sig_y**2
        b -= np.sin(2. * theta) / 4. / sig_x**2
        c = np.sin(theta)**2 / 2. / sig_x**2
        c += np.cos(theta)**2 / 2. / sig_y**2
        kernel = c * grid[:, None]**2 + a * grid[None, :]**2
        kernel += 2 * b * grid[:, None] * grid[None, :]
        return np.exp(-kernel) / 2. / np.pi / sig_x / sig_y

    def convolveZeroth(self, zeroth, beam_maj, beam_min=None, beam_pa=0.0):
        """Convolve the zeroth moment, quicker than whole cube."""
        
        if beam_min is None:
            beam_min = beam_maj
        elif beam_min > beam_maj:
            beam_tmp = beam_min
            beam_min = beam_maj
            beam_maj = beam_tmp
            
        try:
            return self.convolved_zeroths[beam_maj, beam_min, beam_pa]
        except:
            print 'Convolving the zeroth moment.'
            
        beam_kernel = self.beamKernel(beam_maj, beam_min, beam_pa)
        conv_zeroth = convolve(zeroth, beam_kernel)
        self.convolved_zeroths[beam_maj, beam_min, beam_pa] = conv_zeroth
        return self.convolved_zeroths[beam_maj, beam_min, beam_pa]


    def convolveCube(self, beam_maj, beam_min=None, beam_pa=0.0):
        """Use the astropy.convolve package to convolve the cube with a beam.
        """
        
        # Check if the major and minor axes are defined correctly.
        # TODO: Check the convention of minor and major.
        if beam_min is None:
            beam_min = beam_maj
        elif beam_min > beam_maj:
            beam_tmp = beam_min
            beam_min = beam_maj
            beam_maj = beam_tmp
        
        # If the cube has not been convolved with that beam before, run the
        # convolution. Takes a long time...
        try:
            return self.convolved_cubes[beam_maj, beam_min, beam_pa]
        except:
            print 'Convolving new cube.'
            print 'Will take a while. Be patient.'
         
        # Add the convolved cube to the dictionary for easy access later.
        beam_kernel = self.beamKernel(beam_maj, beam_min, beam_pa)
        cube = np.array([convolve(chan, beam_kernel) for chan in self.data])
        self.convolved_cubes[beam_maj, beam_min, beam_pa] = cube
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
            if type(beam) is float:
                beam = [beam, None, 0.0]
            data = self.convolveCube(beam[0], beam[1], beam[2])
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

    
    def getZeroth(self, low=0, high=-1, removeCont=1,
                  bunit=None, vunit='km/s', mask=True, beam=None):
        """Returns the zeroth moment map, integrated between lowchan and
        highchan. If removeCont is set, we remove the continuum, while bunit
        and vunit specify the units. Providing beam parameters will used a
        cube convolved with those parameters. If quick_convolve is set to True, 
        will just convolve the zeroth moment, rather than each channel."""
        
        velo = self.clipVelo(low=low, high=high, vunit=vunit)
        data = self.clipData(low=low, high=high, removeCont=removeCont,
                             beam=beam)
        data *= self.convertBunit(bunit, beam=beam)
        
        zeroth = np.trapz(data, dx=abs(np.diff(velo)[0]), axis=0)
        if (not beam is None and self.quick_convolve):
            if type(beam) is float:
                beam = [beam, None, 0.0]
            elif len(beam) != 3:
                raise ValueError('beam = [b_maj, b_min, b_pa].')
            zeroth = self.convolveZeroth(zeroth, beam[0], beam[1], beam[2])
        if mask:
            return zeroth * self.getNaNMask()
        else:
            return zeroth


    def getFirst(self, low=0, high=-1, removeCont=1,
                 method=1, vunit='km/s', mask=True, beam=None):
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
        if not hasattr(self, 'nanmask'):
            mask = np.sum(self.data, axis=0)
            self.nanmask = np.where(mask == mask.min(), np.nan, 1)
        return self.nanmask

    # Get the radial profile of the zeroth moment.
    def getZerothProfile(self, bins=None, nbins=50, lowchan=0,
                         highchan=-1, removeCont=1, bunit=None,
                         vunit='km/s', mask=True, beam=None):
        if bins is None:
            bins = np.linspace(0, 1.2*self.posax.max(), nbins+1)
        else:
            nbins = len(bins)-1

        ridxs = np.digitize(self.rvals, bins).ravel()
        zeroth = self.getZeroth(low=lowchan, high=highchan,
                                removeCont=removeCont, bunit=bunit,
                                vunit=vunit, mask=mask, beam=beam).ravel()

        avg = np.array([np.nanmean(zeroth[ridxs == r])
                        for r in range(1, nbins+1)])
        std = np.array([np.nanstd(zeroth[ridxs == r])
                        for r in range(1, nbins+1)])
        rad = np.average([bins[1:], bins[:-1]], axis=0)
        return np.array([rad, avg, std])

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

    def convertBunit(self, bunit, beam=None):
        """Convert the brightness units from self.bunit to bunit."""
        
        if (self.bunit == bunit or bunit is None):
            scale = 1.
        elif ('mjy' in bunit.lower() and not 'beam' in bunit.lower()):
            scale = 1e3 / self.JanskytoKelvin()
        elif ('jy' in bunit.lower() and not 'beam' in bunit.lower()):
        
            scale = self.JanskytoKelvin()**-1
        elif 'k' in bunit.lower():
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
