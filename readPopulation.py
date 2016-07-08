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

        self.npts  = int(npts)
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

        self.absorpcoeff = None
        self.absorpcoe_i = None
        self.absorpcoe_j = None

        self.raytrace = None
        self.raytra_i = None
        self.raytra_j = None

        self.maxz = None
        self.minz = None
        self.maxr = None
        self.minr = None
        self.rotz = None
        self.minc = None

        return



    ## Gridding Functions ##

    # TODO: If the bins are empty, prune them.
    # gridcolumn - grid the given column with the given inclination.

    def calcGridColumn(self, col, inc=0.):
        return griddata((self.r_proj(inc), self.z_proj(inc)), self.data[int(col)],
                       (self.rgrid_proj(inc)[None,:], self.zgrid_proj(inc)[:,None]),
                       method='nearest')

    def getGridColumn(self, col, inc=0.):
        return self.clipArray(self.calcGridColumn(col, inc=inc), inc=inc)

    def getBinMax(self, arr, idxs, i, inc=0):
        try:
            maxval = arr[idxs == i].max()
        except ValueError:
            maxval = 0.
        return maxval

    def getBinMin(self, arr, idxs, i, inc=0):
        try:
            minval = arr[idxs == i].min()
        except ValueError:
            minval = 0.
        return minval

    def getModelBounds(self, inc=0.):
        if self.maxz is None:

            idxs = np.digitize(self.r_proj(inc), self.centerstoedges(self.rgrid_proj(inc)))
            self.maxz = np.array([[self.getBinMax(self.z_proj(inc), idxs, i, inc=inc)
                                   for i in np.arange(1, self.npts+1)]])
            self.minz = np.array([[self.getBinMin(self.z_proj(inc), idxs, i, inc=inc)
                                   for i in np.arange(1, self.npts+1)]])

            idxs = np.digitize(self.z_proj(inc), self.centerstoedges(self.zgrid_proj(inc)))
            self.maxr = np.array([[self.getBinMax(self.r_proj(inc), idxs, i, inc=inc)
                                   for i in np.arange(1, self.npts+1)]])
            self.minr = np.array([[self.getBinMin(self.r_proj(inc), idxs, i, inc=inc)
                                   for i in np.arange(1, self.npts+1)]])
            self.minc = [inc]

        elif inc not in self.minc:
            idxs = np.digitize(self.r_proj(inc), self.centerstoedges(self.rgrid_proj(inc)))
            maxz = np.array([self.getBinMax(self.z_proj(inc), idxs, i, inc=inc)
                             for i in np.arange(1, self.npts+1)])
            minz = np.array([self.getBinMin(self.z_proj(inc), idxs, i, inc=inc)
                             for i in np.arange(1, self.npts+1)])
            self.maxz = np.vstack([self.maxz, [maxz]])
            self.minz = np.vstack([self.minz, [minz]])

            idxs = np.digitize(self.z_proj(inc), self.centerstoedges(self.zgrid_proj(inc)))
            maxr = np.array([self.getBinMax(self.r_proj(inc), idxs, i, inc=inc)
                             for i in np.arange(1, self.npts+1)])
            minr = np.array([self.getBinMin(self.r_proj(inc), idxs, i, inc=inc)
                             for i in np.arange(1, self.npts+1)])
            self.maxr = np.vstack([self.maxr, [maxr]])
            self.minr = np.vstack([self.minr, [minr]])
            self.minc.append(inc)

        return self.minr[self.minc.index(inc)], self.maxr[self.minc.index(inc)], self.minz[self.minc.index(inc)], self.maxz[self.minc.index(inc)]

    def clipArray(self, arr, inc=0., mask=np.nan):
        minr, maxr, minz, maxz = self.getModelBounds(inc=inc)

        rarr = self.rgrid_proj(inc)[None,:] * np.ones(self.npts)[:,None]
        arr = np.where(rarr > maxr[:,None]*np.ones(self.npts)[None,:], mask, arr)
        arr = np.where(rarr < minr[:,None]*np.ones(self.npts)[None,:], mask, arr)

        zarr = self.zgrid_proj(inc)[:,None] * np.ones(self.npts)[None,:]
        arr = np.where(zarr > maxz[None,:]*np.ones(self.npts)[:,None], mask, arr)
        arr = np.where(zarr < minz[None,:]*np.ones(self.npts)[:,None], mask, arr)
        return arr

    ## Rotation Functions ##

    # Functions to help with rotated models. inc is always in radians.

    def r_proj(self, inc):
        return self.r * np.cos(inc) - self.z * np.sin(inc)

    def z_proj(self, inc):
        if inc == 0:
            return self.z
        else:
            return self.r * np.sin(inc) + self.z * np.cos(inc)

    def rgrid_proj(self, inc):
        if inc == 0.:
            return self.rgrid
        else:
            r = self.r_proj(inc)
            return np.linspace(-abs(r).max(), abs(r).max(), self.npts)

    def zgrid_proj(self, inc):
        if inc == 0.:
            return self.zgrid
        else:
            z = self.z_proj(inc)
            return np.linspace(-abs(z).max(), abs(z).max(), self.npts)



    ## 2D Physical Structure Functions ##

    # Functions to grid the 2D physical strucutre given an inclination. For fast
    # ways to plot, see Convenience Functions. All can be returned in log_10.

    def getDensity(self, log=False, inc=0.):
        if self.densities is None:
            self.densities = np.array([self.getGridColumn(3, inc=inc)])
            self.density_i = [inc]
        elif inc not in self.density_i:
            self.densities = np.vstack([self.densities, [self.getGridColumn(3, inc=inc)]])
            self.density_i.append(inc)
        if log:
            return np.log10(self.densities[self.density_i.index(inc)])
        else:
            return self.densities[self.density_i.index(inc)]

    def getKineticTemp(self, log=False, inc=0.):
        if self.kinetictemp is None:
            self.kinetictemp = np.array([self.getGridColumn(4, inc=inc)])
            self.kineticte_i = [inc]
        elif inc not in self.kineticte_i:
            self.kinetictemp = np.vstack([self.kinetictemp, [self.getGridColumn(4, inc=inc)]])
            self.kineticte_i.append(inc)
        if log:
            return np.log10(self.kinetictemp[self.kineticte_i.index(inc)])
        else:
            return self.kinetictemp[self.kineticte_i.index(inc)]

    def getRelAbundance(self, log=False, inc=0.):
        if self.relabund is None:
            self.relabund = np.array([self.getGridColumn(5, inc=inc)])
            self.relabu_i = [inc]
        elif inc not in self.relabu_i:
            self.relabund = np.vstack([self.relabund, [self.getGridColumn(5, inc=inc)]])
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
        return self.weightedprofile(self.getGridColumn(int(c)), self.getAbundance(), sampling=sampling)



    ## Level Population Functions ##

    # Functions dealing with the level populations. Note that getLevelRatios
    # will return (n_l * g_u / n_u * g_l). All J levels are specificed with the
    # lower quantum number.

    def getLevelPopulations(self, J, inc=0):
        if self.levelpops is None:
            self.levelpops = np.array([self.getGridColumn(int(7+J), inc=inc)])
            self.levelpo_i = [inc]
            self.levelpo_j = [J]
        idx = np.where(np.logical_and(self.levelpo_i == np.array(inc),
                                      self.levelpo_j == np.array(J)))
        idx = np.squeeze(idx)
        if idx.size == 0:
            self.levelpops = np.vstack([self.levelpops,
                                        [self.getGridColumn(int(7+J), inc=inc)]])
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
        return self.clipArray(a/b, mask=0., inc=inc)

    def getExcitationTemp(self, J, inc=0.):
        if self.excitationtemp is None:
            self.excitationtemp = np.array([calcExcitationTemp(J, inc=inc)])
            self.excitationte_i = [inc]
            self.excitationte_j = [J]
        idx = np.where(np.logical_and(self.excitationte_i == np.array(inc),
                                      self.excitationte_j == np.array(J)))
        idx = np.squeeze(idx)
        if idx.size == 0:
            self.excitationtemp = np.vstack([self.excitationtemp, [calcExcitationTemp(J, inc=inc)]])
            self.excitationte_i.append(inc)
            self.excitationte_j.append(J)
            idx = np.where(np.logical_and(self.excitationte_i == np.array(inc),
                                          self.excitationte_j == np.array(J)))
            idx = np.squeeze(idx)
        return self.excitationtemp[idx]

    def calcExcitationTemp(self, J, inc=0.):
        Tex = sc.h * self.LAMDA.frequencies[J]
        Tex /= sc.k * np.log(self.getLevelRatios(J, inc=inc))
        return Tex

    def getLevelWeightedProfile(self, c, J, sampling=1):
        return self.weightedprofile(self.getGridColumn(int(c)), self.getLevelAbundance(J), sampling=sampling)



    ## Radiative Transfer Functions ##

    # Functions to calculate radtiative transfer properites. Solve basic radiative transfer problem.

    def getEmissionCoeff(self, J, inc=0.):
        emiss = sc.h * self.LAMDA.frequencies[J] * self.LAMDA.EinsteinA[J]
        emiss *= self.getPhi(self.LAMDA.frequencies[J], inc=inc) / 4. / np.pi
        emiss *= self.getLevelAbundance(J+1, inc=inc) * 1e6
        return self.clipArray(emiss, inc=inc, mask=0.)

    def getAbsorptionCoeff(self, J, inc=0.):
        if self.absorpcoeff is None:
            self.absorpcoeff = np.array([self.calcAbsorptionCoeff(J, inc=inc)])
            self.absorpcoe_i = [inc]
            self.absorpcoe_j = [J]
        idx = np.where(np.logical_and(self.absorpcoe_i == np.array(inc),
                                      self.absorpcoe_j == np.array(J)))
        idx = np.squeeze(idx)
        if idx.size == 0:
            self.absorpcoeff = np.vstack([self.absorpcoeff, [self.calcAbsorptionCoeff(J, inc=inc)]])
            self.absorpcoe_i.append(inc)
            self.absorpcoe_j.append(J)
        idx = np.where(np.logical_and(self.absorpcoe_i == np.array(inc),
                                  self.absorpcoe_j == np.array(J)))
        idx = np.squeeze(idx)
        return self.absorpcoeff[idx]

    def calcAbsorptionCoeff(self, J, inc=0.):
        alpha = self.LAMDA.EinsteinA[J] * self.LAMDA.weights[J+1]
        alpha *= self.getPhi(self.LAMDA.frequencies[J], inc=inc)
        alpha *= sc.c**2 / 8. / np.pi / self.LAMDA.weights[J]
        alpha *= self.getLevelAbundance(J, inc=inc) * 1e6
        alpha /= self.LAMDA.frequencies[J]**2.
        alpha *= (1. - 1. / self.getLevelRatios(J, inc=inc))
        return self.clipArray(alpha, mask=0., inc=inc)

    def getTau(self, J, inc=0., negtau=True):
        tau = self.getAbsorptionCoeff(J, inc=inc)
        tau *= (self.zgrid[1]-self.zgrid[0]) * sc.au
        tau = self.clipArray(tau, mask=0., inc=inc)
        if not negtau:
            return np.where(tau < 0., 0., tau)
        else:
            return tau

    def getPhi(self, freq, avgmu=2.34, inc=0.):
        width = np.sqrt(2. * sc.k * self.getKineticTemp(inc=inc) / self.LAMDA.mu / sc.m_p)
        width *= freq / sc.c
        return self.normgaussian(0., 0., width)

    def getSourceFunction(self, J, inc=0.):
        S = self.getEmissionCoeff(J, inc=inc) / self.getAbsorptionCoeff(J, inc=inc)
        return np.where(np.isfinite(S), S, 0.)

    def getCumTau(self, J, direction='down', inc=0.):
        tau = self.getTau(J, inc=inc)
        if direction is 'down':
            ctau = np.array([[np.nansum(tau[j:,i]) for i in range(int(self.npts))]
                              for j in range(int(self.npts))])
        elif direction is 'up':
            ctau = np.array([[np.nansum(tau[:j,i]) for i in range(int(self.npts))]
                              for j in range(int(self.npts))])
        else:
            raise ValueError
        return ctau

    def getTotalIntensity(self, J, inc=0.):
        return self.getRayTrace(J, inc=inc)[-1]

    def getCellIntensity(self, J, inc=0.):
        I = self.getSourceFunction(J, inc=inc)
        I *= (1. - np.exp(-1. * self.getTau(J, inc=inc)))
        return self.clipArray(I, inc=inc)

    def getFluxWeights(self, J, inc=0.):
        totalI = self.getTotalIntensity(J, inc=inc)
        totalI = totalI[None,:] * np.ones(self.npts)[:,None]
        cellI = self.getCellIntensity(J, inc=inc)
        cellI *= np.exp(-1. * self.getCumTau(J, inc=inc, direction='down'))
        fval = cellI / totalI
        return self.clipArray(fval, inc=inc, mask=0.)

    def getRayTrace(self, J, inc=0.):
        if self.raytrace is None:
            self.raytrace = np.array([self.calcRayTrace(J, inc=inc)])
            self.raytra_i = [inc]
            self.raytra_j = [J]
        idx = np.where(np.logical_and(self.raytra_i == np.array(inc),
                                      self.raytra_j == np.array(J)))
        idx = np.squeeze(idx)
        if idx.size == 0:
            self.raytrace = np.vstack([self.raytrace, [self.calcRayTrace(J, inc=inc)]])
            self.raytra_i.append(inc)
            self.raytra_j.append(J)
        idx = np.where(np.logical_and(self.raytra_i == np.array(inc),
                                      self.raytra_j == np.array(J)))
        idx = np.squeeze(idx)
        return self.raytrace[idx]

    def calcRayTrace(self, J, inc=0.):
        t = self.getTau(J, inc=inc)
        S = self.getSourceFunction(J, inc=inc)
        I = np.zeros(t.shape)
        for i in range(self.npts):
            for j in range(self.npts):
                if j > 0:
                    a = I[j-1,i] * np.exp(-1. * t[j,i])
                else:
                    a = 0.
                I[j,i] = a + (1. - np.exp(-1. * t[j,i])) * S[j,i]
        return I

    def getSourceWeightedProfile(self, c, J):
        return self.weightedprofile(self.gridcolumn(int(c)), self.getAbundance()*self.getSourceFunction(J))


    # Find the radial flux weighted profile.
    def getFluxWeightedProfile(self, c, J, dV=None, mach=0., taulim=1.0):
        return self.weightedprofile(self.gridcolumn(int(c)), self.getFluxWeights(J, dV=dV, mach=mach, taulim=taulim))

    ## Miscellaneous Functions ##

    # Miscellaneous functions to help with other functions.

    def normgaussian(self, x, mu, sig):
        return np.exp(-0.5 * np.power((x - mu) / sig, 2.)) / sig / np.sqrt(np.pi * 2.)

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


    ## Convenience Functions ##

    # Functions which also return the projected grids to help with plotting
    # rotated models.

    def plotOutline(self, inc=0.):
        return self.rgrid_proj(inc), self.zgrid_proj(inc), np.where(np.isfinite(self.getDensity(inc=inc)), 1, 0)

    def plotDensity(self, log=True, inc=0.):
        return self.rgrid_proj(inc), self.zgrid_proj(inc), self.getDensity(log=log, inc=inc)

    def plotAbundance(self, log=True, inc=0.):
        return self.rgrid_proj(inc), self.zgrid_proj(inc), self.getAbundance(log=log, inc=inc)

    def plotKineticTemp(self, log=False, inc=0.):
        return self.rgrid_proj(inc), self.zgrid_proj(inc), self.getKineticTemp(log=log, inc=inc)

    def plotRelAbundance(self, inc=0.):
        return self.rgrid_proj(inc), self.zgrid_proj(inc), self.getRelAbundance(inc=inc)

    def plotLevelAbundance(self, J, log=False, inc=0.):
        return self.rgrid_proj(inc), self.zgrid_proj(inc), self.getLevelAbundance(J, log=log, inc=inc)
