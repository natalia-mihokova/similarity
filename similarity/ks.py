"""
the KS test driver

Read the introduction of Hodges (1958), "The significance probability of the Smirnov two-sample test"
P1 is the significance of the

The Kolmogorov (1933) - Smirnov (1939) test is a rank-order test of a two-sample problem,
where one sample may be assumed to be significantly larger than the other (or not!).

This software is primarily focused on the two-sample test, but can be readily repurposed to perform a one-sample style test.

The null hypothesis is that the two samples are equivalent. Given a binomial coefficient of possible orderings (n1+n2, n2),
each has the same probability, 1/(n1+n2,n).
[compute a binomial coefficient in python as scipy.special.binom(a,b)]

When has a distribution strayed too far from the other such that they cannot be consistent?


Follow Section 4 of Hodges for nearly equal sample sizes.

See also Chapter 26 of DasGupta (2008), Aymptotic Theory of Statistics
Much of the theory here is built on the Empirical Distribution Function (EDF)


One must map the KS statistic to a P value, using a KS distribution.
The P value is the answer to this question:

If the two samples were randomly sampled from identical populations, what is the probability that the two cumulative frequency distributions would be as far apart as observed? More precisely, what is the chance that the value of the Komogorov-Smirnov D statistic would be as large or larger than observed?

If the P value is small, conclude that the two groups were sampled from populations with different distributions. The populations may differ in median, variability or the shape of the distribution.

pros:
-The test is nonparametric. It does not assume that data are sampled from Gaussian distributions (or any other defined distributions).
-The results will not change if you transform all the values to logarithms or reciprocals or any transformation. The KS test report the maximum difference between the two cumulative distributions, and calculates a P value from that and the sample sizes. A transformation will stretch (even rearrange if you pick a strange transformation) the X axis of the frequency distribution, but cannot change the maximum distance between two frequency distributions.

cons:
The primary shortcoming in this approach is that we are not particularly sensitive to changes in the tails of the distributions.



"""

import numpy as np

class KS():

    def __init__(self,y1,y2):

        n1,n2 = len(y1),len(y2)

        if n1 != n2:
            print('similarity.ks.KS: mismatched sizes!');return

        self.y1 = y1
        self.y2 = y2

        self.n1 = n1
        self.n2 = n2

        # hard check for bad samples
        #print("ks.KS: n1={}, n2={}".format(self.n1,self.n2))
        if ((self.n1 == 0) | (self.n2==0)):
            print("similarity.ks.KS: invalid samples.")
            self.ks     = np.nan
            self.p1_bar = np.nan
            self.ksDp   = np.nan
            self.ksDm   = np.nan
            self.alpha  = np.nan
            self.kz     = np.nan
            return

        # compute all statistics
        self._compute_ks()     # two-sided test
        self._compute_ks_one() # one-sided test
        self._p1_bar()         # p1 bar
        self._compute_alpha()  # significance level

    def _compute_ks(self):
        """compute the formal two-sided KS test

        see https://www.itl.nist.gov/div898/handbook/eda/section3/eda35g.htm

        in Hodges, this is 'D'.
        The null hypothesis is that the distributions are identical; nonzero values indicate they are not identical (with varying certainty).

        The mathematical operation is called the 'supremum'.
        """

        ks = 0.
        for i in range(1,self.n1):
            testval = np.nanmax([self.y1[i]-self.y2[i-1],self.y2[i]-self.y1[i]])
            if testval > ks:
                ks = testval

        self.ks = ks

    def _compute_ks_one(self):
        """compute the formal one-sided KS test

        in Hodges, this is 'D+' and 'D-'.


        These are the one-sided tests. They examine the null hypotheses that
        D+: y1(x) >= y2(x) for all x; the alternative is that y1(x) < y2(x) for at least one x.
        D-: y2(x) >= y1(x) for all x; the alternative is that y2(x) < y1(x) for at least one x.
        """

        ks = 0.
        for i in range(1,self.n1):
            testval = np.nanmax([self.y1[i]-self.y2[i-1]])
            if testval > ks:
                ks = testval

        self.ksDp = ks

        ks = 0.
        for i in range(1,self.n1):
            testval = np.nanmax([self.y2[i]-self.y1[i]])
            if testval > ks:
                ks = testval

        self.ksDm = ks

    def _p1_bar(self):
        """
        Hodges eq. 5.1, an estimate for fluctuations when the samples are nearly equal (or equal!)

        The remainder here is of order 1/n

        Implement Hodges 5.3 as well.

        This is the significance value we should be targeting to clear in order to disprove the null hypothesis.
        This is a combinatorial approach: we also have empirical calibrations.
        """

        z = self.n1/self.n2

        # Hodges 5.1
        #self.p1_bar = np.exp(-2*z*z - np.sqrt(2)*z/np.sqrt(self.n2))

        # Hodges 5.1
        self.p1_bar = np.exp(-2*z*z - (2*z/3.)*((self.n1+2.*self.n2)/np.sqrt(self.n1*self.n2*(self.n1+self.n2))))

    def _compute_alpha(self):
        """find the significance value, alpha, based on the ks two-sample test

        the null hypothesis is rejected at percentage alpha if
        ks > c(alpha) sqrt((n1+n2)/(n1*n2)) -> c(alpha) sqrt(2/n1)

        [assuming n1=n2]

        where
        c(alpha) = sqrt( -ln(alpha/2)/2 )


        """

        self.alpha = 2.*np.exp(-2.*(self.ks/np.sqrt(2./self.n1))**2.)


    def _smirnovs_variables(self):
        """
        following Hodges Section 2, define some variables

        eq. 2.6 of Hodges

        this reduces to _h_lam if done wisely! (I think)
        and the sample size factors are taken into account
        """

        bigZ    = np.sqrt((n1*n2)/(n1+n2))*self.ks
        zplus   = np.sqrt((n1*n2)/(n1+n2))*self.ksD

        self.kz = 1. - 2.*(np.exp(-2.*(1.*bigZ)**2)-
                         np.exp(-2.*(2.*bigZ)**2)+
                         np.exp(-2.*(3.*bigZ)**2)-
                         np.exp(-2.*(4.*bigZ)**2)+
                         np.exp(-2.*(5.*bigZ)**2)-
                         np.exp(-2.*(6.*bigZ)**2))

    def _h_lam(self,lam,imax=1000):
        """return H(t), the cdf of the KS distribution, for significance value to test, lamda.

        in the limit where n1,n2 -> \infty,

        P_{H_0}\left(\sqrt{n1*n2/(n1+n2)}D_{n1,n2}\le \lambda = P( supremum |B(t)| \le \lambda)

        B(t) is a Brownian bridge, which is a whole thing in and of itself!

        In the large sample limit, we can invert this to find out how likely we are to reject the null hypothesis
        e.g. if we want a p-value of 0.05, then the critical value of D_n = 1.35/sqrt(n)

        """

        ivals = np.arange(1,imax,1)
        sumval = np.nansum( -1.**(ivals-1)*np.exp(-2*ivals*ivals*lam*lam))
        return (1.-2.*sumval)/np.sqrt(self.n1)
