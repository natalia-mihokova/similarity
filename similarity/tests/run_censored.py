"""
this file tests working with censored data (to be extended)

"""

import numpy as np
import pkg_resources

# bring in the package itself
import similarity


from similarity import censored

sample1 = np.array([30,24,11,19,27,11,24,28])
observed1 = np.array([1,0,1,0,1,1,1,0])
sample2 = np.array([3,23,17,8,10,5])
observed2 = np.array([1,1,0,0,1,0])



censored.calculate_twosample(sample1,sample2,observed1,observed2,censoring='right',verbose=1)
