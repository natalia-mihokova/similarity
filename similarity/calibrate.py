"""
calibration of the KS test

for a single distribution
1. subsample distribution against itself

for two distributions:
1. slide distributions against each other for minimum
2. test different grid sizes for interpolation error

"""

import numpy as np

from . import empirical
from . import ks

class SingleTest():
    def __init__(self,Set1,retall=False):

        self.y1 = Set1
        self.n1 = Set1.size

        self.retall = retall

        # create a distribution
        self.D1 = empirical.Distribution(self.y1)

        self._subsample()

    def _subsample(self,nfracs=10,floor_threshold=0.6):
        """a test of discreteness noise in the numbers

        the resulting distributions will decrease monotonically until they hit a floor.
        the scatter about the floor is an estimate in the uncertainty owing to the discreteness in the distribution.

        this process is a bit slow: can we speed it up somehow through code organisation?

        the results are also weirdly unstable? may need more spacing, or to reconsider the combinatorial aspects.
        """

        sample_percentiles = np.linspace(0.1,1.,nfracs)
        ks_values          = np.zeros(nfracs)

        for indx,nfrac in enumerate(sample_percentiles):
            subsample = self.y1[np.random.randint(0,self.n1,int(np.floor(nfrac*self.n1)))]

            # sort...
            #subsample = subsample[subsample.argsort()]

            # create a distribution
            D2 = empirical.Distribution(subsample)

            # compute a homogeneous grid between the two, based on the original spacing.
            empirical.match_distributions(self.D1,D2,spacing=self.D1.dcx)

            # compute the ks values for the two distributions
            K = ks.KS(self.D1.hcdf,D2.hcdf)
            ks_values[indx] = K.ks

        # define the threshold for the noise floor, conservatively 60% of the overall sample size
        criterion = (sample_percentiles>=floor_threshold)

        floor_value   = np.nanmedian(ks_values[criterion])
        floor_scatter =    np.nanstd(ks_values[criterion])

        self.uncertainty_subsample = floor_scatter

        if self.retall:
            self.percentiles = sample_percentiles
            self.ks_values   = ks_values

        


class DoubleTest():
    def __init__(self,Set1,Set2,verbose=0):

        self.y1 = Set1
        self.y2 = Set2
        self.verbose = verbose

        # create the distributions
        self.D1 = empirical.Distribution(self.y1)
        self.D2 = empirical.Distribution(self.y2)

        # get our baseline
        empirical.match_distributions(self.D1,self.D2)

        # compute the KS statistic(s)
        K = ks.KS(self.D1.hcdf,self.D2.hcdf)
        self.base_ks = K.ks

        self._test_spacing()
        self._test_shift()

    def _test_spacing(self):

        base_spacing = np.nanmax([self.D1.dcx,self.D2.dcx])

        spacing_range = np.array([0.1,0.125,0.25,0.5,1,2,4,8])
        ks_values     = np.zeros(spacing_range.size)
        
        for indx,multiplier in enumerate(spacing_range):
            empirical.match_distributions(self.D1,self.D2,spacing=base_spacing*multiplier)

            K = ks.KS(self.D1.hcdf,self.D2.hcdf)
            ks_values[indx] = K.ks

        self.uncertainty_spacing = np.nanstd(ks_values)

        # only for some serious debugging
        if self.verbose>1:
            print('calibrate.DoubleTest._test_spacing: max uncertainty at spacing={0:4.3f}, KS={0:4.3f}'.format(spacing_range[np.nanargmax(ks_values)],np.nanmax(ks_values)))
        

    def _test_shift(self):
        """
        find the minimum KS value when offsetting the distributions relative to one another.
        
        DOES NOT APPLY TO ALL SITUATIONS!

        """

        # establish a maximum shift: very generous here!
        base_shift = np.abs(np.nanmedian(self.y1) - np.nanmedian(self.y2))
        
        max_shift = np.nanmax([np.abs(np.nanpercentile(self.y1,75) - np.nanpercentile(self.y2,25)),
                               np.abs(np.nanpercentile(self.y2,75) - np.nanpercentile(self.y1,25))])

        # how to test for multiple minima?

        # first step, just quote global minumum

        shift_vals = np.linspace(-max_shift,max_shift,30)
        ks_values  = np.zeros(shift_vals.size)
        
        for indx,shift in enumerate(shift_vals):
            # shift all values of D2 relative to D1
            D2s = empirical.Distribution(self.y2 - shift)
            empirical.match_distributions(self.D1,D2s)

            K = ks.KS(self.D1.hcdf,D2s.hcdf)
            ks_values[indx] = K.ks

        self.uncertainty_shifting = np.nanmin(ks_values)


