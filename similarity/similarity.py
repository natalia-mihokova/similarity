"""
the similarity score wrapper

more reading:
- https://www.itl.nist.gov/div898/handbook/eda/section3/eda35g.htm
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html#scipy.stats.kstest
- https://en.wikipedia.org/wiki/Sign_test
- https://en.wikipedia.org/wiki/Logistic_regression

"""

import numpy as np

from . import empirical
from . import ks

class compare():
    """
    """
    def __init__(self,Set1,Set2):
        """
        given two different sets of values, compute the similarity score and
        estimates of associated uncertainties

        this wrapper will use all defaults under the hood -- you may want to access the classes directly for your application.

        inputs
        --------
        Set1
        Set2

        """

        self.similarity = 0.
        self.significance = 0.

        D1 = empirical.Distribution(Set1)
        D2 = empirical.Distribution(Set2)

        # homogenise to share grids
        empirical.match_distributions(D1,D2)

        # compute the KS statistic(s)
        K = ks.KS(D1.hcdf,D2.hcdf)

        # transfer to the compare namespace
        self.ks = K.ks
        self.ksDp = K.ksDp
        self.ksDm = K.ksDm
        self.p1_bar = K.p1_bar
        
