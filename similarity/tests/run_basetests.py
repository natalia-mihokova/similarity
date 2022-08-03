"""
this file tests the basic KS implementation with some example data

"""
import numpy as np
import pkg_resources
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# test the base classes? not needed unless implementing new features
base_class_test = False

# bring in the package itself
import similarity

# identify the testing files
d1 = pkg_resources.resource_filename('similarity','data/Donaldson/LMC.evolved.heliocentric.solar.system1_3eb.dat')
d2 = pkg_resources.resource_filename('similarity','data/Donaldson/MW.evolved.heliocentric.solar.system1_5eb.dat')

# read in the models, which are 6d positions of dark matter particles in the solar neighbourhood
model1 = np.genfromtxt(d1,delimiter=';',skip_header=1)
model2 = np.genfromtxt(d2,delimiter=';',skip_header=1)

# make speed distributions
vtot1 = np.sqrt(model1[:,3]*model1[:,3] + model1[:,4]*model1[:,4] + model1[:,5]*model1[:,5])
vtot2 = np.sqrt(model2[:,3]*model2[:,3] + model2[:,4]*model2[:,4] + model2[:,5]*model2[:,5])

# now bring in the default measurement techniques
from similarity import similarity

# run the base comparison between two samples: the two-sample KS test.
#   this will generate three KS statistics (two-sided, greater, less [compare to scipy])
#   and return a p-value for significance (here, alpha)
result = similarity.compare(vtot1,vtot2)
print("Summary statistics for the two distributions:")
print("KSD={},KSD+={},KSD-={},pvalue={}".format(result.ks,result.ksDp,result.ksDm,result.p1_bar))

# do a comparison to scipy?
from scipy.stats import ks_2samp

res = ks_2samp(vtot1,vtot2,mode='exact')
print("Scipy comparison:")
print("KSD={},pvalue={}".format(res.statistic,res.pvalue))
# the p-values don't quite match here -- why not?

# test the calibration classes
print("\n Moving to calibration exercises...")
from similarity import calibrate

# uncertainties intrinsic to one distribution
# if we were missing some fraction of observations, how does the KS value change?
C = calibrate.SingleTest(vtot1)
print('Uncertainty from subsampling distribution 1:',C.uncertainty_subsample)

C = calibrate.SingleTest(vtot2)
print('Uncertainty from subsampling distribution 2:',C.uncertainty_subsample)
# WARNING: these values are not repeatable. We might need to consider even more samples


# uncertainties intrinsic to two distributions
C = calibrate.DoubleTest(vtot1,vtot2,verbose=1)

# if we used different bins to construct the CDFs, how much uncertainty do we gain?
print('Uncertainty from constructing matched CDFs:',C.uncertainty_spacing)

# test the hypothesis that the distributions might just be shifted relative to each other:
# what is the minimum KS value we can get simply by shifting?
print('Minimum KS value when shifting CDFs:',C.uncertainty_shifting)



# bring in some different analytical distributions
from similarity.analytical import *

# now we'd like to test the standard halo model
velvals = np.linspace(0.,800,1000) # in km/s
vdisp = 117

# realise a 1d Maxwellian velocity distribution (would work if dimensions were symmetric!)
maxwellian_shm = Maxwellian(velvals,vdisp)

# realise a 3d gaussian with (U,V,W) velocity offsets applied for the solar neighborhood
shm = MultiGaussian(velvals)
shm.trivariate_gaussian_pdf(vdisp*np.sqrt(3),cen1=11.1,cen2=229.+12.24,cen3=7.25)


# make a plot?
plt.figure(figsize=(4,3))

plt.plot(maxwellian_shm.xvals,maxwellian_shm.pdf,label='Maxwellian',color='grey')
plt.plot(shm.xvals,shm.pdf,label='SHM',color='black')
plt.xlabel('velocity (km/s)')
plt.ylabel('pdf')
plt.legend()

plt.tight_layout()
plt.savefig('testSHM.png')

# test the unwrapped functions
if base_class_test == True:

    # from similarity.empirical, make sure we can match distributions
    from similarity import empirical

    D1 = empirical.Distribution(vtot1)
    D2 = empirical.Distribution(vtot1*1.0001)

    empirical.match_distributions(D1,D2)

    # from similarity.ks, compute additional statistics from Hodges (1958)
    from similarity import ks

    K = ks.KS(D1.hcdf,D2.hcdf)

    print(K.ks)
    #K._significance()
    print(K.alpha)
    print('p1bar',K.p1_bar)

    print(K._h_lam(0.07,imax=1000))
    #print('hlam',K.p1_bar)
