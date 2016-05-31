# analyseLIME
Scripts to help analyse LIME output.


## outputCLASS.py

All functions are designed to work with standard LIME output.
  
#### Grids

`getVelocityAxes()` - Returns the velocity axis.
    
`getPositionAxes()` - Returns the on-sky position axis.
    
`getSpectralAxes()` - Returns the frequency axis.
    
`getModelAxes()` - Returns the deprojected model axes.
    
`getRadialPoints()` - Returns the deprojected radial position of each pixel in arcseconds.
    
`getPolarPoints()` - Returns the polar angle of each pixel in radians.
  
  
#### Units

`JanskytoKelvin()` - Returns the factor to convert Jy to K.


#### Momemnts

`getZeroth(withCont=False)` - Returns the integrated intensity. If withCont=False, will use continuum subtracted data.

`getFirst(withCont=False)` - Returns the intensity weighted mean velocity. If withCont=False, will use continuum subtracted data.

`getSecond(withCont=False)` - Returns the intensity weighted velocity dispersion. If withCont=False, will use continuum subtracted data.
    
`getThird(withCont=False)` - Returns the intensity weighted skewness. If withCont=False,will use continuum subtracted data. Uses a natural method of moments estimator of the population skewness [1].


#### Continuum

`getContinuum()` - Returns the continuum calcualted from the first channel.
    
`removeContiuum()` - Returns data with the continuum subtracted. Note this is a simple subtraction, only the first channel is removed from all channels. Make sure the first channel is free of line emission.
    

#### Radial Profiles
 
`getIntensityProfile(toavg, bins=None, edges=None, percentiles=None)` - Returns an azmuthally averaged profile of the provided array. Bin centres or edges can be specified for the binning. The default is 50 radial points spanning from 0 to r_max. If percentiles are specified, returns the request percentiles, if not specified, will return the average and standard deviation.


#### Miscellaneous
 
`swapedgescenters(centers=None, edges=None)` -Swaps between bin centers and edges. 


---
##### References
[1] - Joanes, D. N.; Gill, C. A. (1998). "Comparing measures of sample skewness and kurtosis". Journal of the Royal Statistical Society (Series D): The Statistician 47 (1): 183–189. doi:10.1111/1467-9884.00122.
