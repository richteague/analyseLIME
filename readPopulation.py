import numpy as np
from analyseLIME import readLAMDA as rates
import scipy.constants as sc
from scipy.interpolate import griddata

class population:
    
    ## Initialisation ##
    
    # On initialisation, read in the popfile and collisional rates.
    # Automatically prune non-disk points and convert (x,y,z) to [au].
    # Change the density to [cm^3].
    
    def __init__(self, popfile, collisionalrates, npts=200, verbose=True,
                 turbulence=50., turbunit='absolute'):
        
        self.verbose = True
        
        # Read in the data. Tries both standard ASCII input
        # and .npy files. Will transpose the array if necessary
        # to have the parameters in the zeroth dimension.
        
        try:
            self.data = np.loadtxt(popfile)
        except ValueError:
            self.data = np.load(popfile)
        if self.data.shape[0] > self.data.shape[1]:
            self.data = self.data.T
        
        if np.nanmean(np.log10(self.data[0])) > 6.:
            if verbose:
                print 'Assuming (x,y,z) in [m]. Converting to [au].'
            self.data[:3] /= sc.au

        if verbose:
                print 'Assuming n(H2) in [m^-3]. Converting to [cm^-3].'
        self.data[3] /= 1e6
        
        # Remove all points which aren't part of the disk.
        # This is done only where there is non-zero abundance.
        
        if verbose:
                print 'Read in population file with %d points.' % (self.data.shape[1])
        self.data = np.array([self.data[i][self.data[5] > 0]
                              for i in range(self.data.shape[0])])
        if verbose:
                print 'Pruned to %d disk points.' % (self.data.shape[1])
        
        # Read in the LAMDA collisional rate file.
        # This is used to grab frequencies, energies and Einstein As.
        
        self.LAMDA = rates.ratefile(collisionalrates)
        self.LAMDAfilename = collisionalrates
        if verbose:
                print 'Attached rates for %s.' % self.LAMDA.molecule
        
        
        # Dictionaries and variables for analysis.
        self.turbulence = turbulence
        self.turbunit = turbunit
        self.npts = npts
        self.xgrid = self.getgrids(self.npts)[0]
        self.ygrid = self.getgrids(self.npts)[1]
        self.r = np.hypot(self.data[0], self.data[1])
        self.z = abs(self.data[2])
               
        self.excitationtemp = {}
        self.levelabundance = {}
        self.levelratios = {}
        self.phi = {}
        self.sourcefunction = {}
        self.absorptioncoeff = {}
        self.tau = {}
        self.cumulativetau = {}
        self.cellflux = {}
        self.totalflux = {}
        self.fluxweights = {}
        
        return
    
    
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

    
    # Calculate the surface density.
    def getSurfaceDensity(self, unit='ccm', trim=False):
        toint = self.getDensity()
        toint = np.where(np.isnan(toint), 0., toint).T
        sigma = np.array([np.trapz(col, x=self.ygrid*sc.au*100.)
                          for col in toint])
        if unit == 'gccm':
            sigma *= sc.m_p * 2. * 1e3 * 0.85
        elif unit == 'kgccm':
            sigma *= sc.m_p * 2. * 0.85
        if trim:
            return np.array([self.xgrid[sigma > 0], 2.*sigma[sigma > 0]])
        else:
            return np.array([self.xgrid, 2.*sigma])
    
    
    # Calculate the molecular column density.
    def getColumnDensity(self, unit='ccm', trim=False):
        toint = self.getAbundance()
        toint = np.where(np.isnan(toint), 0., toint)
        sigma = np.array([np.trapz(col, x=self.ygrid*sc.au*100.) 
                          for col in toint.T])
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
        if not self.levelabundance.has_key(J):
            if self.verbose:
                print 'Gridding level population J = %d.' % J
            self.levelabundance[J] = self.gridcolumn(int(7+J)) 
            self.levelabundance[J] *= self.getRelAbundance() * self.getDensity()
        return self.levelabundance[J]

    
    
    # Get the excitation temperature of a given level.
    def getExcitationTemp(self, J):
        if not self.excitationtemp.has_key(J):
            Tex = sc.h * self.LAMDA.frequencies[J] / sc.k
            Tex /= np.log(self.getLevelRatios(J))
            self.excitationtemp[J] = Tex
        return self.excitationtemp[J]
    
    
    # Get the ratio of level populations, including weights.
    # Returns: n_l * g_u / n_u / g_l
    def getLevelRatios(self, J):
        if not self.levelratios.has_key(J):
            a = self.getLevelAbundance(J) * self.LAMDA.weights[J+1] 
            b = self.getLevelAbundance(J+1) * self.LAMDA.weights[J]
            self.levelratios[J] = a / b
        return self.levelratios[J]
    
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
    
    
    # Return the level populated radial profile.
    def getLevelWeightedProfile(self, c, J):
        return self.weightedprofile(self.gridcolumn(int(c)), 
                                    self.getLevelAbundance(J))
    
    
    # Find the radial, abundance weighted profile for a given parameter in column c.
    def getAbundanceWeightedProfile(self, c):
        return self.weightedprofile(self.gridcolumn(int(c)), 
                                    self.getAbundance())
    
    
    # Find the radial flux weighted profile.
    def getFluxWeightedProfile(self, c, J):
        return self.weightedprofile(self.gridcolumn(int(c)), 
                                    self.getFluxWeights(J))
    

    # Get the line width.
    def getPhi(self, J):
        if not self.phi.has_key(J):
            width = 2. * sc.k * self.getKineticTemp()
            width /= self.LAMDA.mu * sc.m_p
            vturb = self.turbulence**2.
            if self.turbunit in 'mach':
                vturb *= sc.k * self.getKineticTemp()
                vturb /= self.LAMDA.mu * sc.m_p
            elif not self.turbunit in 'absolute':
                raise ValueError("turbunit must be either 'absolute' or 'mach'.")
            width = np.sqrt(width+vturb) * self.LAMDA.frequencies[J] / sc.c
            phi = 1. / width / np.sqrt(2. * np.pi)
            self.phi[J] = np.where(np.isfinite(phi), phi, 0)
        return self.phi[J]
    
    # Calculate the absorption coefficient.
    def getAbsorptionCoeff(self, J):
        if not self.absorptioncoeff.has_key(J):
            alpha = self.LAMDA.EinsteinA[J] * self.LAMDA.weights[J+1] 
            alpha *= sc.c**2 / 8. / np.pi / self.LAMDA.weights[J]
            alpha *= self.getLevelAbundance(J) * 1e6 / self.LAMDA.frequencies[J]**2.
            alpha *= (1. - 1./self.getLevelRatios(J))
            alpha *= self.getPhi(J)
            self.absorptioncoeff[J] = np.where(np.isfinite(alpha), alpha, 0.0)
        return self.absorptioncoeff[J]
    
    # Calculate the source function.
    def getSourceFunction(self, J):
        if not self.sourcefunction.has_key(J):
            S = 2. * sc.h * self.LAMDA.frequencies[J]**3 / sc.c**2
            S /= (self.getLevelRatios(J) - 1.)
            self.sourcefunction[J] = np.where(np.isfinite(S), S, 0.)
        return self.sourcefunction[J]
    
    # Calculate the optical depth of each cell. 
    def getTau(self, J):
        if not self.tau.has_key(J):
            dz = np.diff(self.ygrid)[0] * sc.au
            tau = self.getAbsorptionCoeff(J) 
            self.tau[J] = np.where(np.isfinite(tau), tau*abs(dz), 0.)
        return self.tau[J]
    
    # Calculate the cumulative flux.
    def getCumulativeTau(self, J):
        if not self.cumulativetau.has_key(J):
            tau = self.getTau(J)
            ctau = np.array([[np.nansum(tau[j:,i]) for i in range(self.npts)] 
                             for j in range(self.npts)])
            self.cumulativetau[J] = np.where(np.isfinite(ctau), ctau, 0.0)
        return self.cumulativetau[J]
    
    # Total flux vertically integrated.
    def getTotalFlux(self, J):
        if not self.totalflux.has_key(J):
            tau = self.getTau(J)
            phi = self.getPhi(J)
            Snu = self.getSourceFunction(J)
            emission = np.zeros(tau.shape[1])
            for i in range(1,self.npts):
                dI = Snu[i] * (1. - np.exp(-1. * tau[i]))
                emission += np.where(np.isfinite(dI), dI, 0.0)
                emission = np.where(emission > 0.0, emission, 0.0)
            self.totalflux[J] = emission
        return self.totalflux[J]
    
    # Flux from each cell.
    def getCellFlux(self, J):
        if not self.cellflux.has_key(J):
            tau = self.getTau(J)
            phi = self.getPhi(J)
            Snu = self.getSourceFunction(J)
            self.cellflux[J] = Snu * (1. - np.exp(-tau))
        return self.cellflux[J]
        
    # Flux weights.
    def getFluxWeights(self, J):
        if not self.fluxweights.has_key(J):
            fw = self.getCellFlux(J) 
            fw *= np.exp(-self.getCumulativeTau(J))
            #fw /= self.getTotalFlux(J)[None,:]
            fw /= np.sum(fw, axis=0)
            self.fluxweights[J] = np.where(np.log10(fw) >= -10, fw, 0.)
        return self.fluxweights[J]
        
