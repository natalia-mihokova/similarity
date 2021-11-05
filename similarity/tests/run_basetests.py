"""
this file tests the basic distributions

"""
import numpy as np
import pkg_resources

# bring in the package itself
import similarity

# identify the testing files
d1 = pkg_resources.resource_filename('similarity','data/LMC.evolved.galactocentric.solar.system1_3eb.dat')
d2 = pkg_resources.resource_filename('similarity','data/LMC.evolved.galactocentric.solar.system1_5eb.dat')

# read in the models, which are 6d positions of dark matter particles in the solar neighbourhood
model1 = np.genfromtxt(d1,delimiter=';',skip_header=1)
model2 = np.genfromtxt(d2,delimiter=';',skip_header=1)

# now bring in the default measurement
from similarity import similarity

vtot1 = np.sqrt(model1[:,3]*model1[:,3] + model1[:,4]*model1[:,4] + model1[:,5]*model1[:,5])
vtot2 = np.sqrt(model2[:,3]*model2[:,3] + model2[:,4]*model2[:,4] + model2[:,5]*model2[:,5])

result = similarity.compare(vtot1,vtot2)
print(result.ks,result.ksDp,result.ksDm,result.p1_bar)


from similarity import empirical

D1 = empirical.Distribution(vtot1)
D2 = empirical.Distribution(vtot1*1.0001)

#print(D1.cdf)

empirical.match_distributions(D1,D2)

# test the calibration classes
from similarity import calibrate

C = calibrate.SingleTest(vtot1)
print('uncertainty from subsampling 1:',C.uncertainty_subsample)

C = calibrate.SingleTest(vtot2)
print('uncertainty from subsampling 2:',C.uncertainty_subsample)

C = calibrate.DoubleTest(vtot1,vtot2,verbose=1)
print('baseline KS',C.base_ks)
print('uncertainty from constructing matched CDFs:',C.uncertainty_spacing)
print('minimum value when shifting CDFs:',C.uncertainty_shifting)

# test the raw KS classes
from similarity import ks

K = ks.KS(D1.hcdf,D2.hcdf)

print(K.ks)
#K._significance()
print(K.alpha)
print('p1bar',K.p1_bar)

print(K._h_lam(0.07,imax=1000))
#print('hlam',K.p1_bar)

# bring in some different analytical distributions
from similarity.analytical import *

velvals = np.linspace(0.,800,1000)
vdisp = 117

maxwellian_shm = Maxwellian(velvals,vdisp)

shm = MultiGaussian(velvals)
shm.trivariate_gaussian_pdf(vdisp*np.sqrt(3),-100.)
shm.trivariate_gaussian_pdf(vdisp*np.sqrt(3),cen1=11.1,cen2=229.+12.24,cen3=7.25)

#mcdf = make_cdf(velvals,maxwellian(velvals,vdisp))
#plt.plot(maxwellian_shm.xvals,maxwellian_shm.pdf)
#plt.plot(shm.xvals,shm.pdf)




from similarity import censored

sample1 = np.array([30,24,11,19,27,11,24,28])
observed1 = np.array([1,0,1,0,1,1,1,0])
sample2 = np.array([3,23,17,8,10,5])
observed2 = np.array([1,1,0,0,1,0])



censored.calculate_twosample(sample1,sample2,observed1,observed2,censoring='right',verbose=1)

