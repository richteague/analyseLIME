# Class to read in LIME files and analyse them quickly.

import numpy as np
from astropy.io import fits
from astropy.io.fits import getval
import scipy.constants as sc

class LIMEoutput:


    def __init__(self, file, inc, dist=None):
    
        # self.filename - filename of the LIME output.
        # self.inc - inclination in radians of the source.
        
        # Data and axes.
        self.filename = file
        self.data = fits.getdata(self.filename, 0)
        self.velax = self.getVelocityAxes()
        self.posax = self.getPositionAxes()
        
        # Commonly used values.
        self.inc = inc
        self.dist = dist
        self.x0 = getval(self.filename, 'crpix1', 0) - 1.
        self.y0 = getval(self.filename, 'crpix2', 0) - 1.
        self.bunit = fits.getval(self.filename, 'bunit', 0)
        
        # Flags to speed up repeat calculations.
        self.zeroth_last = None
        self.first_last = None
        self.second_last = None
        self.third_last = None


     ## Functions ##
    


    def getModelAxes(self):
        return self.posax, self.posax/np.cos(self.inc)
    
    def getIntensityProfile(self, toavg, bins=None, edges=None, percentiles=None):
        rvals = self.getRadialPoints().ravel()
        if bins is None and edges is None:
            redges = np.linspace(0., rvals.max(), 51)
        elif bins is not None and edges is None:
            redges = self.swapedgescenters(centers=bins)
        else:
            redges = edges
        ridxs = np.digitize(rvals, redges)
        rad = self.swapedgescenters(edges=redges)
        if percentiles is None:
            avg = np.array([np.nanmean(toavg.ravel()[ridxs == r]) for r in range(1, redges.size)])
            std = np.array([np.nanstd(toavg.ravel()[ridxs == r]) for r in range(1, redges.size)])
            return np.array([rad, avg, std])
        else:
            pct =  np.array([np.percentile(toavg.ravel()[ridxs == r], percentiles) for r in range(1, redges.size)]).T            
            return np.insert(pct, 0, rad, axis=0)
            
        
    def getContinuum(self):
        if not hasattr(self, 'contperchan'):
            self.contperchan = self.data[0] - self.data[0,0,0]
        return self.contperchan
    
    
    def removeContinuum(self):
        if not hasattr(self, 'contsubdata'):
            self.contsubdata = np.array([chan - self.getContinuum() for chan in self.data])
        return self.contsubdata
    
    
    def getVelocityAxes(self):
        a_len = getval(self.filename, 'naxis3', 0)
        a_del = getval(self.filename, 'cdelt3', 0)
        a_pix = getval(self.filename, 'crpix3', 0)    
        a_ref = getval(self.filename, 'crval3', 0)
        velax = ((np.arange(1, a_len+1) - a_pix) * a_del) + a_ref
        if getval(self.filename, 'cunit3') is not 'm/s':
            velax * sc.c / getval(self.filename, 'restfreq')
        return velax
        
        
    def getPositionAxes(self):
        a_len = getval(self.filename, 'naxis2', 0)
        a_del = getval(self.filename, 'cdelt2', 0)
        a_pix = getval(self.filename, 'crpix2', 0)    
        a_ref = getval(self.filename, 'crval2', 0)        
        return 3600.*(((np.arange(1, a_len+1) - a_pix) * a_del) + a_ref)
        
        
    def getSpectralAxes(self): 
        return self.getVelocityAxes() * getval(self.filename, 'restfreq') / sc.c
        
        
    def JanskytoKelvin(self):
        jy2k = 10**-26 * sc.c**2. / 2. / fits.getval(self.filename, 'restfreq', 0)**2.
        jy2k /= sc.k * np.radians(fits.getval(self.filename, 'cdelt2', 0))**2.
        return jy2k
        
        
    def getZeroth(self, withCont=False):
        if not (hasattr(self, 'zeroth') and self.zeroth_last == withCont):
            print 'Calculating zeroth moment.'
            if withCont:
                tempdata = self.data
            else:
                tempdata = self.removeContinuum()
            self.zeroth = np.array([[np.trapz(tempdata[:,j,i], x=self.velax)
                                     for i in range(tempdata.shape[2])]
                                     for j in range(tempdata.shape[1])])
            self.zeroth_last = withCont
        return self.zeroth
        
        
    def getFirst(self, withCont=False):
        if not (hasattr(self, 'first') and self.first_last == withCont):
            print 'Calculating first moment.'
            if withCont:
                tempdata = self.data
            else:
                tempdata = self.removeContinuum()
            self.first = np.array([[np.average(self.velax, weights=tempdata[:,j,i])
                                     for i in range(tempdata.shape[2])]
                                     for j in range(tempdata.shape[1])])
            self.first_last = withCont
        return self.first   
    
    
    def getSecond(self, withCont=False):
        if not (hasattr(self, 'second') and self.second_last == withCont):
            print 'Calculating second moment.'
            if withCont:
                tempdata = self.data
            else:
                tempdata = self.removeContinuum()
            self.second = np.array([[np.average((self.velax-self.getFirst(withCont=withCont)[j,i])**2.,
                                                 weights=tempdata[:,j,i])**0.5
                                     for i in range(tempdata.shape[2])]
                                     for j in range(tempdata.shape[1])])
            self.second_last = withCont        
        return self.second


    def getThird(self, withCont=False):
        if not (hasattr(self, 'third') and self.third_last == withCont):
            print 'Calculating third moment.'
            if withCont:
                tempdata = self.data
            else:
                tempdata = self.removeContinuum()
            skew = np.array([[np.sum((self.velax - self.getFirst(withCont=withCont)[j,i])**3.) / self.velax.size
                              for i in range(tempdata.shape[2])]
                              for j in range(tempdata.shape[1])])    
            skew /= np.array([[(np.sum((self.velax - self.getFirst(withCont=withCont)[j,i])**2.) / (self.velax.size - 1.))**1.5
                                for i in range(tempdata.shape[2])]
                                for j in range(tempdata.shape[1])])
            self.third = skew
            self.third_last = withCont
        return self.third
        
        
    def swapedgescenters(self, centers=None, edges=None):
        if centers is not None:
            edg = np.average([centers[:-1], centers[1:]], axis=0)
            edg = np.insert(edg, 0, 2.*centers[0]-edg[0])
            edg = np.append(edg, 2.*centers[-1]-edg[-1])
            return edg
        if edges is not None:
   	    return np.average([edges[1:], edges[:-1]], axis=0) 
    
    def getRadialPoints(self):
        if  not hasattr(self, 'rvals'):
            xvals = self.getModelAxes()[0][None,:] * np.ones(self.posax.size)[:,None]
            yvals = self.getModelAxes()[1][:,None] * np.ones(self.posax.size)[None,:]
            self.rvals = np.hypot(xvals, yvals/np.cos(self.inc))
        return self.rvals
        
        
    def getPolarPoints(self):
        if  not hasattr(self, 'tvals'):
            xvals = (np.arange(self.data.shape[2]) - self.x0)[None,:] * np.ones(self.data.shape[1])[:,None]
            yvals = (np.arange(self.data.shape[1]) - self.y0)[:,None] * np.ones(self.data.shape[2])[None,:]
            self.tvals = np.arctan2(yvals/np.cos(self.inc), xvals)
        return self.tvals 
        
