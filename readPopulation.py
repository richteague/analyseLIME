import numpy as np
import readRateFile as rates
import scipy.constants as sc
from scipy.interpolate import griddata

class population:
    
    ## Initialisation ##
    
    # On initialisation, read in the popfile and collisional rates.
    # Automatically prune non-disk points and convert (x,y,z) to [m].
    # Change the density to [cm^3].
    
    def __init__(self, popfile, collisionalrates, npts=200):
        
        # Read in the data. Tries both standard ASCII input
        # and .npy files. Will transpose the array if necessary
        # to have the parameters in the zeroth dimension.
        
        try:
            self.data = np.loadtxt(popfile)
        except ValueError:
            self.data = np.load(popfile)
        if self.data.shape[0] > self.data.shape[1]:
            self.data = self.data.T
        
        # Unit conversions. Will try and determine if the
        # units are in [au] or [m]. If in [m] will convert
        # to [au]. Similarly will automatically assume that
        # the input is in [m^-3] and covert to [cm^-3].
        
        if np.nanmean(np.log10(self.data[0])) > 6.:
            print 'Assuming (x,y,z) in [m]. Converting to [au].'
            self.data[:3] /= sc.au

        print 'Assuming n(H2) in [m^-3]. Converting to [cm^-3].'
        self.data[3] /= 1e6

        
        # Remove all points which aren't part of the disk.
        # This is done only where there is non-zero abundance.
        
        print 'Read in population file with %d points.' % (self.data.shape[1])
        self.data = np.array([self.data[i][self.data[5] > 0]
                              for i in range(self.data.shape[0])])
        print 'Pruned to %d disk points.' % (self.data.shape[1])
        
        # Read in the LAMDA collisional rate file.
        # This is used to grab frequencies, energies and Einstein As.
        
        self.LAMDA = rates.ratefile(collisionalrates)
        self.LAMDAfilename = collisionalrates
        print 'Attached rates for %s.' % self.LAMDA.molecule
        
        # Set up some standard holders to be filled in later.
        # self.npts      - number of points used for gridding.
        # self.levelpops - gridded J populations.
        # self.leveltaus - gridded taus for each transition.
        
        # Get the grids.
        self.npts = npts
        self.xgrid = self.getgrids(self.npts)[0]
        self.ygrid = self.getgrids(self.npts)[1]
        self.r = np.hypot(self.data[0], self.data[1])
        self.z = abs(self.data[2])
        
        # Holders for flags 
        self.lastdV = None
        self.lastmach = None
        self.lastfreq = None
        self.lastavgmu = None
        
        self.lastJ_int = None
        self.lastdV_int = None
        self.lastmach_int = None
        
        self.fluxweight = None
        self.lastJ_fweight = None
        self.lastdV_fweight = None
        self.lastmach_fweight = None
        self.lasttaulim_fweight = None
        
        self.levelpops = np.zeros((self.data.shape[0]-6, npts, npts))
        self.leveltaus = np.zeros((self.data.shape[0]-6, npts, npts))
        
        
        
        return
    
    
    
    
    ### Functions ###
    

    
    ## Gridding Functions ##
    
    # Calculate the grids.
    def getgrids(self, npts):
        xgrid = np.linspace(0, np.hypot(self.data[0], self.data[1]).max(), npts)
        ygrid = np.linspace(-abs(self.data[2]).max(), abs(self.data[2]).max(), npts) 
        return xgrid, ygrid
    
    
    # Switch between bin edges and centers for binning.
    def swapedgescenters(self, centers=None, edges=None):
        if centers is not None:
            edg = np.average([centers[:-1], centers[1:]], axis=0)
            edg = np.insert(edg, 0, 2.*centers[0]-edg[0])
            edg = np.append(edg, 2.*centers[-1]-edg[-1])
            return edg
        if edges is not None:
            cnt = np.zeros(edges.size-1)
            cnt[0] = edges[:2].mean()
            for i in range(1, edges.size-1):
                cnt[i] = edges[i+1]-cnt[i-1]
            return cnt

    
    # Two dimensional gridding.
    def gridcolumn(self, col):
        arr = griddata((np.hypot(self.data[0], self.data[1]), self.data[2]), 
                        self.data[int(col)], 
                        (self.xgrid[None,:], self.ygrid[:,None]),
                        method='nearest', fill_value=0.)
        if not hasattr(self, 'template'):
            self.template = griddata((np.hypot(self.data[0], self.data[1]), self.data[2]),
                                      self.data[4],
                                     (self.xgrid[None,:], self.ygrid[:,None]),
                                     method='cubic', fill_value=np.nan)
        return np.where(np.isnan(self.template), np.nan, arr)
    
    
    # Get 2D histogram of LIME points.
    def getpIntensities(self):
        H, _, _ = np.histogram2d(np.hypot(self.data[0], self.data[1]), abs(self.data[2]),
                                 bins=[self.centerstoedges(self.xgrid), 
                                       self.centerstoedges(self.ygrid)])
        return H.T
    
    
    
    ## Structure Functions ##
    
    # Calculate the surface density.
    def getSurfaceDensity(self, unit='ccm', trim=False):
        toint = self.getDensity()
        toint = np.where(np.isnan(toint), 0., toint)
        sigma = np.array([np.trapz(col, x=self.getyGrid()*sc.au*100.) for col in toint.T])
        if unit == 'gccm':
            sigma *= sc.m_p * 2. * 1e3 * 0.85
        elif unit == 'kgccm':
            sigma *= sc.m_p * 2. * 0.85
        if trim:
            return np.array([self.getxGrid()[sigma > 0], 2.*sigma[sigma > 0]])
        else:
            return np.array([self.getxGrid(), 2.*sigma])
    
    
    # Calculate the molecular column density.
    def getColumnDensity(self, unit='ccm', trim=False):
        toint = self.getAbundance()
        toint = np.where(np.isnan(toint), 0., toint)
        sigma = np.array([np.trapz(col, x=self.ygrid*sc.au*100.) for col in toint.T])
        if unit == 'gccm':
            sigma *= sc.m_p * self.LAMDA.mu * 1e3
        elif unit == 'kgccm':
            sigma *= sc.m_p * self.LAMDA.mu
        if trim:
            return np.array([self.xgrid[sigma > 0], 2.*sigma[sigma > 0]])
        else:
            return np.array([self.xgrid, 2.*sigma])

    
    # Return the collider density structure.
    def getDensity(self):
        if not hasattr(self, 'density'):
            print 'Gridding main collider density.'
            self.density = self.gridcolumn(3)
        return self.density
    
    
    # Return the kinetic temperature structure.
    def getKineticTemp(self):
        if not hasattr(self, 'tkin'):
            print 'Gridding %s kinetic temperature.' % self.LAMDA.molecule
            self.tkin = self.gridcolumn(4)
        return self.tkin
    
    
    # Return the molecular realtive abundance structure.
    def getRelAbundance(self):
        if not hasattr(self, 'relabund'):
            print 'Gridding %s relative abundance.' % self.LAMDA.molecule
            self.relabund = self.gridcolumn(5)
        return self.relabund
    
    
    # Get the outline of the gridding for pretty plotting.
    def getOutline(self):
        if not hasattr(self, 'outline'):
            self.outline = np.where(np.isfinite(self.getDensity()), 1, 0)
        return self.outline
    
    
    # Return the molecular abundance structure.
    def getAbundance(self):
        if not hasattr(self, 'abund'):
            self.abund = self.getRelAbundance() * self.getDensity()
        return self.abund
    
    
    # Return the level abundance of the Jth level.
    def getLevelAbundance(self, J):
        if np.nansum(self.levelpops[J]) == 0:
            print 'Gridding level population J = %d.' % J
            self.levelpops[J] = self.gridcolumn(int(7+J))
        return self.levelpops[J] * self.getRelAbundance() * self.getDensity()
    
    
    # Get the excitation temperature of a given level.
    def getExcitationTemp(self, J):
        self.Tex = sc.h * self.LAMDA.frequencies[J] / sc.k
        self.Tex /= np.log(self.getLevelRatios(J))
        self.lastTexJ = J
        return self.Tex
    
    
    # Get the ratio of level populations, including weights.
    # Returns: n_l * g_u / n_u / g_l
    def getLevelRatios(self, J):
        a = self.getLevelAbundance(J) * self.LAMDA.weights[J+1] 
        b = self.getLevelAbundance(J+1) * self.LAMDA.weights[J]
        return a/b
    

    
    ## Radial Profile Functions ##
    
    # Calculate the weighted profile.
    def weightedprofile(self, params, weight):    
        params = np.where(np.isfinite(params), params, 0.)
        weight = np.where(np.isfinite(weight), weight, 0.)
        idx = np.array([r for r in range(weight.shape[1]) 
                        if np.sum(weight[:,r]) > 0.])
        avg = np.array([np.average(params[:,r], weights=weight[:,r]) 
                        for r in idx])
        std = np.array([np.average((params[:,rr]-avg[r])**2. / params[:,rr].size, weights=weight[:,rr]) 
                        for r, rr in enumerate(idx)])**0.5
        return np.array([self.xgrid[idx], avg, std])
    
    
    # Get source weighted averaged (similar to abundance weighted.)
    def getSourceWeightedProfile(self, c, J):
        return self.weightedprofile(self.gridcolumn(int(c)), self.getAbundance()*self.getSourceFunction(J))
    
    
    # Return the level populated radial profile.
    def getLevelWeightedProfile(self, c, J):
        return self.weightedprofile(self.gridcolumn(int(c)), self.getLevelAbundance(J))
    
    
    # Find the radial, abundance weighted profile for a given parameter in column c.
    def getAbundanceWeightedProfile(self, c):
        return self.weightedprofile(self.gridcolumn(int(c)), self.getAbundance())
    
    
    # Find the radial flux weighted profile.
    def getFluxWeightedProfile(self, c, J, dV=None, mach=0., taulim=1.0):
        return self.weightedprofile(self.gridcolumn(int(c)), self.getFluxWeights(J, dV=dV, mach=mach, taulim=taulim))
    
    
    
    ## Radiative Transfer Functions ##
    
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
    
    
    # Calculate the cumulative flux.
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
