"""
Kaplan & Meier (1958) product estimator, as described in Feigelson & Nelson (1985).

This formalism was used in Petersen et al. (2019)




# a very simple example
sample1 = np.array([30,24,11,19,27,11,24,28])
observed1 = np.array([1,0,1,0,1,1,1,0])
sample2 = np.array([3,23,17,8,10,5])
observed2 = np.array([1,1,0,0,1,0])



calculate_twosample(sample1,sample2,observed1,observed2,censoring='right',verbose=1)

"""


import numpy as np

from scipy.stats import chi2


    
def two_sample_prep(sample1,sample2,observed1,observed2,censoring='left'):
    '''
    
    inputs
    ------------
    sample1    : numpy array, observed values or upper limits for first sample
    sample2    : numpy array, observed values or upper limits for first sample
    observed1  : numpy array, same size as sample1. each entry is 1 for detections, 0 for upper limits
    observed2  : numpy array, same size as sample2. each entry is 1 for detections, 0 for upper limits
    censoring  : 'left' or 'right', indicating upper or lower limits
    
    '''
    
    if censoring=='left':
        # find the offsetting value (maximum observed value)
        offset_val = np.max([np.max(sample1[observed1==1]),np.max(sample2[observed2==1])])

        sample1 = offset_val-sample1
        sample2 = offset_val-sample2
    else:
        #
        # right censored is the base assumption. 
        #   most survival analysis is written for right censoring, e.g. death during trial
        #
        print('two_sample_prep: Assuming right-censoring (lower limits).')

    # parition data
    true1 = sample1[observed1==1]
    censored1 = sample1[observed1==0]

    true2 = sample2[observed2==1]
    censored2 = sample2[observed2==0]

    # make the joint sample and rankings
    jsample = np.unique(np.concatenate([sample1[observed1==1],sample2[observed2==1]]))
    jranking = jsample.argsort()
    #print(jsample[jranking])

    # calculate basic statistics
    N1 = sample1.size
    N2 = sample2.size
    rank1 = sample1.argsort()
    rank2 = sample2.argsort()
    n = N1 + N2
    r = np.max([N1,N2]) - 1
    #print(N1,N2,n,r)
    
    return jsample,sample1,true1,censored1,sample2,true2,censored2




def calculate_twosample(sample1,sample2,observed1,observed2,censoring='right',test='peto',verbose=0):
    '''
    
    inputs
    ------------
    sample1    : numpy array, observed values or upper limits for first sample
    sample2    : numpy array, observed values or upper limits for first sample
    observed1  : numpy array, same size as sample1. each entry is 1 for detections, 0 for upper limits
    observed2  : numpy array, same size as sample2. each entry is 1 for detections, 0 for upper limits
    censoring  : 'left' or 'right', indicating upper or lower limits
    test       : 'peto','gehan','logrank', indicating which statistical test to undertake
    verbose    : verbosity flag.
    
    
    
    '''
    
    
    jsample,sample1,true1,censored1,sample2,true2,censored2 = two_sample_prep(sample1,sample2,observed1,observed2,censoring=censoring)


    logrank_L_sum = 0
    logrank_sig_sum = 0
    gehan_L_sum = 0
    gehan_sig_sum = 0
    peto_L_sum = 0
    peto_sig_sum = 0

    peto_w = np.zeros(jsample.size)

    for indx,j in enumerate(jsample):

        n1j = len( np.where(sample1 >= j)[0]) 
        n2j = len( np.where(sample2 >= j)[0])
        nj  = n1j + n2j
        peto_w[indx] = nj/(nj+1)


        #if indx+1 < len(jsample):
        #    m1j = len( np.where( (censored1 >= j) & (censored1 < jsample[indx+1]))[0])
        #    m2j = len( np.where( (censored2 >= j) & (censored2 < jsample[indx+1]))[0])
        #    mj  = m1j + m2j

        #else:
        #    m1j = m2j = mj = 0.

        d1j = len( np.where(true1 == j)[0])
        d2j = len( np.where(true2 == j)[0])
        dj  = d1j + d2j

        # calculate the peto-prentice weight
        wj = np.prod(peto_w[0:indx])

        if indx+1 < len(jsample):
            # now use wj * (d1j - dj*n1j/nj)
            #   for gehan, wj= nj
            gehan_L   = nj * (d1j - (dj*n1j/nj))
            gehan_sig = nj*nj*dj * ((n1j*n2j)/(nj*nj)) * ((nj-dj)/(nj-1))


            logrank_L   = (d1j - (dj*n1j/nj))
            logrank_sig = dj * ((n1j*n2j)/(nj*nj)) * ((nj-dj)/(nj-1))

            peto_L = wj * (d1j - (dj*n1j/nj))
            peto_sig = wj*wj*dj * ((n1j*n2j)/(nj*nj)) * ((nj-dj)/(nj-1))

        else:
            gehan_L = gejan_sig = logrank_L = logrank_sig = 0.

        gehan_L_sum += gehan_L
        gehan_sig_sum += gehan_sig

        logrank_L_sum += logrank_L
        logrank_sig_sum += logrank_sig

        peto_L_sum += peto_L
        peto_sig_sum += peto_sig

        #if verbose > 1:
        #    print(j,n1j,n2j,nj,m1j,m2j,mj,d1j,d2j,dj,logrank_L,logrank_sig)

    if verbose > 0:
        print('Logrank:       L={0:6.2f} s={1:6.2f} p={2:5.4f}'.format(logrank_L_sum,logrank_sig_sum,chi2.sf((logrank_L_sum*logrank_L_sum)/logrank_sig_sum,1)))
        print('Peto-Prentice: L={0:6.2f} s={1:6.2f} p={2:5.4f}'.format(peto_L_sum,peto_sig_sum,chi2.sf((peto_L_sum*peto_L_sum)/peto_sig_sum,1)))
        print('Gehan:         L={0:6.2f} s={1:6.2f} p={2:5.4f}'.format(gehan_L_sum,gehan_sig_sum,chi2.sf((gehan_L_sum*gehan_L_sum)/gehan_sig_sum,1)))
        
    if test == 'gehan':
        return chi2.sf((gehan_L_sum*gehan_L_sum)/gehan_sig_sum,1)
    elif test == 'logrank':
        return chi2.sf((logrank_L_sum*logrank_L_sum)/logrank_sig_sum,1)
    else:
        return chi2.sf((peto_L_sum*peto_L_sum)/peto_sig_sum,1)

    
