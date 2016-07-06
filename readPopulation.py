import numpy as np
import readLAMDA as rates
import scipy.constants as sc
from scipy.interpolate import griddata

class population:

    ## Initialisation ##

    # On initialisation, read in the popfile and collisional rates.
    # Automatically prune non-disk points and convert (x,y,z) to [m].
    # Change the density to [cm^3].

    def __init__(self, popfile, collisionalrates, npts=200, prune=True):

        ## Read In ##

        #Tries both standard ASCII input and .npy files. Will transpose the
        #array if necessary to have the parameters in the zeroth dimension.

        try:
            self.data = np.loadtxt(popfile)
        except ValueError:
            self.data = np.load(popfile)
        if self.data.shape[0] > self.data.shape[1]:
            self.data = self.data.T


        ## Unit Conversions #

        # Will try and determine if the units are in [au] or [m]. If in [m] will
        # convert to [au]. Similarly will automatically assume that the input is
        # in [m^-3] and covert to [cm^-3].

        if np.nanmean(np.log10(self.data[0])) > 6.:
            print 'Assuming (x,y,z) in [m]. Converting to [au].'
            self.data[:3] /= sc.au

        print 'Assuming n(H2) in [m^-3]. Converting to [cm^-3].'
        self.data[3] /= 1e6


        ## Pruning ##

        # Remove all points which aren't part of the disk. This is done only
        # where there is non-zero molecular abundance.

        print 'Read in population file with %d points.' % (self.data.shape[1])
        if prune:
            self.data = np.array([self.data[i][self.data[5] > 0]
                                  for i in range(self.data.shape[0])])
            print 'Pruned down to %d disk points.' % (self.data.shape[1])


        ## Collisional Rates ##

        # Read in the LAMDA collisional rate file. See the readLAMDA.py
        # documentation for a description of the options.

        self.LAMDA = rates.ratefile(collisionalrates)
        self.LAMDAfilename = collisionalrates
        print 'Attached rates for %s.' % self.LAMDA.molecule


        ## Gridding ##

        # Work in cylindrical coordinates. Use these values for the whole
        # dataset and also the standard non-inclined structure.

        self.true_r = np.hypot(self.data[0], self.data[1])
        self.true_z = self.data[2]
        self.true_p = np.arctan2(self.data[1], self.data[0])

        self.r = np.sign(self.true_p) * self.true_r
        self.z = self.true_z

        self.npts  = npts
        self.rgrid = np.linspace(-abs(self.r).max(), self.r.max(), self.npts)
        self.zgrid = np.linspace(-abs(self.z).max(), self.z.max(), self.npts)


        ## Arrays ##

        # Create arrays to hold intermediate griddings. This will save time
        # when it comes to having to recalculate values. Each array comes with
        # an associated inclination array, _i, and for level dependent arrays,
        # a level array, _j.

        self.densities = None
        self.density_i = None

        self.kinetictemp = None
        self.kineticte_i = None

        self.relabund = None
        self.relabu_i = None

        self.levelpops = None
        self.levelpo_i = None
        self.levelpo_j = None

        self.excitationtemp = None
        self.excitationte_i = None
        self.excitationte_j = None

        return



    ## Gridding Functions ##

    # gridcolumn - grid the given column with the given inclination.

    def gridcolumn(self, col, inc=0., method='nearest'):
        arr = griddata((self.r_proj(inc), self.z_proj(inc)), self.data[int(col)],
                       (self.rgrid_proj(inc)[None,:], self.zgrid_proj(inc)[:,None]),
                       method=method, fill_value=0.)
        if method is not 'nearest':
            return arr
        else:
            tmp = griddata((self.r_proj(inc), self.z_proj(inc)), self.data[4],
                           (self.rgrid_proj(inc)[None,:], self.zgrid_proj(inc)[:,None]),
                           method='linear', fill_value=np.nan)
            return np.where(np.isnan(tmp), np.nan, arr)



    ## Rotation Functions ##

    # Functions to help with rotated models. inc is always in radians.

    def r_proj(self, inc):
        return self.r * np.cos(inc) - self.z * np.sin(inc)

    def z_proj(self, inc):
        return self.r * np.sin(inc) + self.z * np.cos(inc)

    def rgrid_proj(self, inc):
        r = self.r_proj(inc)
        z = self.z_proj(inc)
        return np.linspace(-abs(r).max(), abs(r).max(), self.npts)

    def zgrid_proj(self, inc):
        r = self.r_proj(inc)
        z = self.z_proj(inc)
        return np.linspace(-abs(z).max(), abs(z).max(), self.npts)



    ## 2D Physical Structure Functions ##

    # Functions to grid the 2D physical strucutre given an inclination. For fast
    # ways to plot, see Convenience Functions. All can be returned in log_10.

    def getDensity(self, log=False, inc=0.):
        if self.densities is None:
            self.densities = np.array([self.gridcolumn(3, inc=inc)])
            self.density_i = [inc]
        elif inc not in self.density_i:
            self.densities = np.vstack([self.densities, [self.gridcolumn(3, inc=inc)]])
            self.density_i.append(inc)
        if log:
            return np.log10(self.densities[self.density_i.index(inc)])
        else:
            return self.densities[self.density_i.index(inc)]

    def getKineticTemp(self, log=False, inc=0):
        if self.densities is None:
            self.kinetictemp = np.array([self.gridcolumn(4, inc=inc)])
            self.kineticte_i = [inc]
        elif inc not in self.density_i:
            self.kinetictemp = np.vstack([self.kinetictemp, [self.gridcolumn(4, inc=inc)]])
            self.kineticte_i.append(inc)
        if log:
            return np.log10(self.kinetictemp[self.kineticte_i.index(inc)])
        else:
            return self.kinetictemp[self.kineticte_i.index(inc)]

    def getRelAbundance(self, log=False, inc=0):
        if self.densities is None:
            self.relabund = np.array([self.gridcolumn(5, inc=inc)])
            self.relabu_i = [inc]
        elif inc not in self.density_i:
            self.relabund = np.vstack([self.relabund, [self.gridcolumn(5, inc=inc)]])
            self.relabu_i.append(inc)
        if log:
            return np.log10(self.relabund[self.relabu_i.index(inc)])
        else:
            return self.relabund[self.relabu_i.index(inc)]

    def getAbundance(self, log=False, inc=0.):
        return self.getRelAbundance(inc=inc) * self.getDensity(log=log, inc=inc)



    ## 1D Physical Structure Functions ##

    # Functions to get only the radial properties (TODO: currently spans +- r).
    # The weighted function will use the self.rgrid as the default binning. This
    # can be coarsened with the sampling parameter (TODO: check errors.).

    def getSurfaceDensity(self, unit='ccm', trim=False):
        toint = self.getDensity(inc=0.)
        toint = np.where(np.isnan(toint), 0., toint)
        sigma = np.array([np.trapz(col, x=self.zgrid*sc.au*100.) for col in toint.T])
        if unit == 'gccm':
            sigma *= sc.m_p * 2. * 1e3 * 0.85
        elif unit == 'kgccm':
            sigma *= sc.m_p * 2. * 0.85
        if trim:
            return np.array([self.rgrid[sigma > 0], 2.*sigma[sigma > 0]])
        else:
            return np.array([self.rgrid, 2.*sigma])

    def getColumnDensity(self, unit='ccm', trim=False):
        toint = self.getAbundance(inc=0)
        toint = np.where(np.isnan(toint), 0., toint)
        sigma = np.array([np.trapz(col, x=self.zgrid*sc.au*100.) for col in toint.T])
        if unit == 'gccm':
            sigma *= sc.m_p * self.LAMDA.mu * 1e3
        elif unit == 'kgccm':
            sigma *= sc.m_p * self.LAMDA.mu
        if trim:
            return np.array([self.rgrid[sigma > 0], 2.*sigma[sigma > 0]])
        else:
            return np.array([self.rgrid, 2.*sigma])

    def getAbundanceWeightedProfile(self, c, sampling=1):
        return self.weightedprofile(self.gridcolumn(int(c)), self.getAbundance(), sampling=sampling)



    ## Level Population Functions ##

    # Functions dealing with the level populations. Note that getLevelRatios
    # will return (n_l * g_u / n_u * g_l). All J levels are specificed with the
    # lower quantum number.

    def getLevelPopulations(self, J, inc=0):
        if self.levelpops is None:
            self.levelpops = np.array([self.gridcolumn(int(7+J), inc=inc)])
            self.levelpo_i = [inc]
            self.levelpo_j = [J]
        idx = np.where(np.logical_and(self.levelpo_i == np.array(inc),
                                      self.levelpo_j == np.array(J)))
        idx = np.squeeze(idx)
        if idx.size == 0:
            self.levelpops = np.vstack([self.levelpops,
                                        [self.gridcolumn(int(7+J), inc=inc)]])
            self.levelpo_i.append(inc)
            self.levelpo_j.append(J)
            idx = np.where(np.logical_and(self.levelpo_i == np.array(inc),
                                          self.levelpo_j == np.array(J)))
            idx = np.squeeze(idx)
        return self.levelpops[idx]

    def getLevelAbundance(self, J, log=False, inc=0.):
         return self.getLevelPopulations(J, inc=inc) * self.getAbundance(log=log, inc=inc)

    def getLevelRatios(self, J, inc=0):
        a = self.getLevelAbundance(J, inc=inc) * self.LAMDA.weights[J+1]
        b = self.getLevelAbundance(J+1, inc=inc) * self.LAMDA.weights[J]
        return a/b

    def getExcitationTemp(self, J, inc=0.):
        if self.excitationtemp is None:
            Tex = sc.h * self.LAMDA.frequencies[J]
            Tex /= sc.k * np.log(self.getLevelRatios(J, inc=inc))
            self.excitationtemp = np.array([Tex])
            self.excitationte_i = [inc]
            self.excitationte_j = [J]
        idx = np.where(np.logical_and(self.excitationte_i == np.array(inc),
                                      self.excitationte_j == np.array(J)))
        idx = np.squeeze(idx)
        if idx.size == 0:
            Tex = sc.h * self.LAMDA.frequencies[J]
            Tex /= sc.k * np.log(self.getLevelRatios(J, inc=inc))
            self.excitationtemp = np.vstack([self.excitationtemp, [Tex]])
            self.excitationte_i.append(inc)
            self.excitationte_j.append(J)
            idx = np.where(np.logical_and(self.excitationte_i == np.array(inc),
                                          self.excitationte_j == np.array(J)))
            idx = np.squeeze(idx)
        return self.excitationtemp[idx]

    def getLevelWeightedProfile(self, c, J, sampling=1):
        return self.weightedprofile(self.gridcolumn(int(c)), self.getLevelAbundance(J), sampling=sampling)



    ## Miscellaneous Functions ##

    # Miscellaneous functions to help with other functions.

    def runningmean(self, data, sampling):
        csum = np.cumsum(np.insert(data, 0, 0))
        return (csum[sampling:] - csum[:-sampling]) / sampling

    def edgestocenters(self, edges):
        return np.average([edges[1:], edges[:-1]], axis=0)

    def centerstoedges(self, centers):
        edg = np.average([centers[:-1], centers[1:]], axis=0)
        edg = np.insert(edg, 0, 2.*centers[0]-edg[0])
        edg = np.append(edg, 2.*centers[-1]-edg[-1])
        return edg

    def weightedprofile(self, params, weight, sampling=1):
        params = np.where(np.isfinite(params), params, 0.)
        weight = np.where(np.isfinite(weight), weight, 0.)
        idx = np.array([r for r in range(weight.shape[1])
                        if np.sum(weight[:,r]) > 0.])
        avg = np.array([np.average(params[:,r], weights=weight[:,r])
                        for r in idx])
        std = np.array([np.average((params[:,rr]-avg[r])**2. / params[:,rr].size, weights=weight[:,rr])
                        for r, rr in enumerate(idx)])**0.5
        rax = self.rgrid[idx]
        if sampling > 1:
            rax = self.runningmean(self.rgrid[idx], sampling)
            avg = self.runningmean(avg, sampling)
            std = self.runningmean(std, sampling)
        return np.array([rax, avg, std])



    ## Radiative Transfer Functions ##

    # Functions to calculate radtiative transfer properites.
    # TODO: lots.


    def getSourceWeightedProfile(self, c, J):
        return self.weightedprofile(self.gridcolumn(int(c)), self.getAbundance()*self.getSourceFunction(J))


    # Find the radial flux weighted profile.
    def getFluxWeightedProfile(self, c, J, dV=None, mach=0., taulim=1.0):
        return self.weightedprofile(self.gridcolumn(int(c)), self.getFluxWeights(J, dV=dV, mach=mach, taulim=taulim))

    # Normalised Gaussian function.
    def normgaussian(self, x, mu, sig):
        return np.exp(-0.5 * np.power((x - mu) / sig, 2.)) / sig / np.sqrt(np.pi * 2.)


    # Calculate the line profile, assuming broadened Guassians..
    def calcPhi(self, dV=None, mach=0., avgmu=2.34, freq=None):
        width = np.sqrt((mach**2 / avgmu + 2. / self.LAMDA.mu) * sc.k * self.getKineticTemp() / sc.m_p)
        if freq is not None:
            width *= freq / sc.c
        if (dV is not None) or (dV == 0.):
            velax = np.linspace(-dV/2., dV/2., 200.)
            if freq is not None:
                velax *= freq / sc.c
            phi = np.zeros(width.shape)
            for i in range(width.shape[1]):
                for j in range(width.shape[0]):
                    if np.isfinite(width[j,i]):
                        phi[j,i] = np.trapz(self.normgaussian(velax, 0., width[j,i]), x=velax)
                    else:
                        phi[j,i] = np.nan
        else:
            phi = self.normgaussian(0., 0., width)
        return phi


    # Return the line profile integrated over some bandwidth in [m/s] assuming some Mach number.
    def getPhi(self, freq=None, dV=None, mach=0., avgmu=2.34):
        if not (hasattr(self, 'phi') and
               (dV == self.lastdV) and
               (mach == self.lastmach) and
               (freq == self.lastfreq) and
               (avgmu == self.lastavgmu)):
            print 'Recalculating phi...'
            self.lastdV = dV
            self.lastmach = mach
            self.lastfreq = freq
            self.lastavgmu = avgmu
            self.phi = self.calcPhi(dV=dV, mach=mach, avgmu=avgmu, freq=freq)
        return self.phi


    # Calculate the absorption coefficient.
    def getAbsorptionCoeff(self, J, dV=None, mach=0.):
        alpha = self.LAMDA.EinsteinA[J] * self.LAMDA.weights[J+1]
        alpha *= sc.c**2 / 8. / np.pi / self.LAMDA.weights[J]
        alpha *= self.getLevelAbundance(J) * 1e6 / self.LAMDA.frequencies[J]**2.
        alpha *= (1. - 1./self.getLevelRatios(J))
        return alpha * self.getPhi(dV=dV, mach=mach, freq=self.LAMDA.frequencies[J])


    # Calculate the source function.
    def getSourceFunction(self, J):
        S = 2. * sc.h * self.LAMDA.frequencies[J]**3 / sc.c**2
        S /= (self.getLevelRatios(J) - 1.)
        return np.where(np.isfinite(S), S, 0.)


    # Calculate the optical depth of each cell.
    def getTau(self, J, dV=None, mach=0.):
        dz = (self.ygrid[1]-self.ygrid[0]) * sc.au
        tau = self.getAbsorptionCoeff(J, dV, mach) * dz
        return np.where(np.isfinite(tau), tau, 0.)


    # Calculate the cumulative optical depth.
    def getcTau(self, J, dV=None, mach=0., direction='down'):
        tau = self.getTau(J, dV, mach)
        if direction == 'up':
            ctau = np.array([[np.sum(tau[j:,i]) for i in range(self.xgrid.size)]
                       for j in range(self.ygrid.size)])
            return np.where(np.isnan(self.template), np.nan, ctau)
        elif direction != 'down':
            print "Wrong direction value: %s.\nAssuming 'down'." % direction
        ctau = np.array([[np.sum(tau[:-j,i]) for i in range(self.xgrid.size)]
                               for j in range(self.ygrid.size)])
        return np.where(np.isnan(self.template), np.nan, ctau)


    # Integrate ray along the column.
    def calcColumn(self, J, dV=None, mach=0.):
        if not (hasattr(self, 'I') and
                (self.lastJ_int == J) and
                (self.lastdV_int == dV) and
                (self.lastmach_int == mach)):
            tau = self.getTau(J, dV=dV, mach=mach)
            Snu = self.getSourceFunction(J)
            I = np.zeros(tau.shape)
            for i in range(self.xgrid.size):
                for j in range(self.ygrid.size):
                    if j > 0:
                        a = I[j-1,i] * np.exp(-1. * tau[j,i])
                    else:
                        a = 0.
                    I[j,i] = a + (1. - np.exp(-1. * tau[j,i])) * Snu[j,i]
            self.I = I
        return self.I[-1,:], np.where(np.isnan(self.template), np.nan, self.I)

    def getFluxWeights(self, J, dV=None, mach=0., taulim=1.):
        if not (hasattr(self, 'fluxweight') and
                (self.lastJ_fweight == J) and
                (self.lastdV_fweight == dV) and
                (self.lastmach_fweight == mach) and
                (self.lasttaulim_fweight == taulim)):
            flux = (1. - np.exp(-1. * self.getTau(J, dV=dV, mach=mach)))
            flux *= self.getSourceFunction(J)
            flux = np.where(self.getcTau(J, dV=dV, mach=mach, direction='down') <= taulim, flux, -0.)
            weight = flux * (1.-np.exp(-1. * self.getcTau(J, dV=dV, mach=mach, direction='up')))
            weight /= np.nansum(weight, axis=0)[None, :]
            self.fluxweight = np.where(np.isnan(self.template), np.nan, weight)
        return self.fluxweight



    ## Convenience Functions ##

    # Functions which also return the projected grids to help with plotting
    # rotated models.

    def plotOutline(self, inc=0.):
        return self.rgrid_proj(inc), self.zgrid_proj(inc), np.where(np.isfinite(self.getDensity(inc=inc)), 1, 0)

    def plotDensity(self, log=True, inc=0.):
        return self.rgrid_proj(inc), self.zgrid_proj(inc), self.getDensity(log=log, inc=inc)

    def plotAbundance(self, log=True, inc=0.):
        return self.rgrid_proj(inc), self.zgrid_proj(inc), self.getAbundance(log=log, inc=inc)

    def plotKineticTemp(self, inc=0.):
        return self.rgrid_proj(inc), self.zgrid_proj(inc), self.getKineticTemp(inc=inc)

    def plotRelAbundance(self, inc=0.):
        return self.rgrid_proj(inc), self.zgrid_proj(inc), self.getRelAbundance(inc=inc)

    def plotLevelAbundance(self, J, log=False, inc=0.):
        return self.rgrid_proj(inc), self.zgrid_proj(inc), self.getLevelAbundance(J, log=log, inc=inc)
