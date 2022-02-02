"""
class structure for some basic 1d analytic distributions

Each distribution comes with a pdf and cdf, sampled at points as specified in the call.
Additional parameters are required as specified for each distribution.

supported distributions:
-Chi^2
-Gaussian
-Lognormal
-Maxwellian
-Multivariate Gaussian -> primarily 3d
-Normal (alias for Gaussian)
-Cauchy
-Lorentzian (alias for Cauchy)

To add more, one may build a new class, which must have the following properties:
-xvals = the 1d sample points
-pdf   = the probability density function
-cdf   = the cumulative density function


todo:
-extend to 2d?

"""

import numpy as np

from scipy.special import erf,gamma,gammainc


class Logistic():
    """the logistic distribution
    the pdf is often used in neural nets
    https://en.wikipedia.org/wiki/Logistic_distribution
    """
    def __init__(self,xvals,mu=0.,s=1.):
        self.xvals = xvals
        self.dx    = xvals[1]-xvals[0]
        self.mu    = mu
        self.s     = s

        self._logistic_pdf()
        self._logistic_cdf()

    def _logistic_pdf(self):
        """the pdf for the logistic distribution"""
        arg = np.exp(-(self.xvals-self.mu)/self.s)

        self.pdf = (arg/(self.s*(1.+arg)**2.))


    def _logistic_cdf(self):
        """the cdf for the logistic distribution"""
        arg = np.exp(-(self.xvals-self.mu)/self.s)

        self.cdf = (1./(1.+arg))



class Lorentzian():
    """alias for Cauchy, if you're a physicist"""
    def __init__(self,xvals,gamma=1.,x0=0.):
        C = Cauchy(xvals,gamma,x0)
        self.xvals = C.xvals
        self.dx    = C.dx
        self.gamma = C.gamma
        self.x0    = C.x0
        self.cdf   = C.cdf
        self.pdf   = C.pdf


class Cauchy():
    """the Cauchy (Lorentz) distribution
    https://en.wikipedia.org/wiki/Cauchy_distribution
    """
    def __init__(self,xvals,gamma=1.,x0=0.):
        self.xvals = xvals
        self.dx    = xvals[1]-xvals[0]
        self.gamma = gamma
        self.x0    = x0

        self._cauchy_pdf()
        self._cauchy_cdf()

    def _cauchy_pdf(self):
        """Cauchy distribution pdf"""
        norm = (np.pi*self.gamma)

        self.pdf = (norm*(1.+((self.xvals-self.x0)/self.gamma)**2))**(-1.)

    def _cauchy_cdf(self):
        """Cauchy distribution cdf"""
        norm = (1./np.pi)

        self.cdf = norm * np.arctan((self.xvals-self.x0)/self.gamma) + 0.5



class Chi2():
    def __init__(self,xvals,k=1):

        self.xvals = xvals
        self.dx    = xvals[1]-xvals[0]
        self.k     = k

        self._chi2_pdf()
        self._chi2_cdf()

    def _chi2_pdf(self):
        """chi2 distribution pdf"""

        norm = 2.**(self.k/2.)*gamma(self.k/2.)

        tpdf = (self.xvals**(self.k/2. - 1) * np.exp(-self.xvals/2.))/norm

        self.pdf = tpdf#/np.nansum(self.dx*tpdf)

    def _chi2_cdf(self):
        """chi2 distribution cdf"""

        norm = gamma(self.k/2.)

        # the scipy definition includes the norm: so don't include here!
        tcdf = gammainc(self.k/2.,self.xvals/2.)#/norm

        self.cdf = tcdf



class Gaussian():
    """the OG. The distribution of a random variable with unknown distribution.

    owes to the central limit theorem: under some conditions, the average of many sample of a random variable,
    with finite mean and variance (which is itself a random variable), will converge to a normal distribution
    as the number of samples increases.

    """
    def __init__(self,xvals,mu=0.,sigma=1.):

        self.xvals = xvals
        self.mu    = mu
        self.sigma = sigma

        self._gaussian_pdf()
        self._gaussian_cdf()

    def _gaussian_pdf(self):
        """gaussian distribution pdf"""
        self.pdf = (1./self.sigma/np.sqrt(2.*np.pi)) * np.exp(-0.5*((self.xvals-self.mu)/self.sigma)**2.)

    def _gaussian_cdf(self):
        """gaussian distribution cdf"""
        self.cdf = (1./2.) * (1.+erf((self.xvals-self.mu)/(np.sqrt(2)*self.sigma)))


class Lognormal():
    def __init__(self,xvals,mu=0.,sigma=1.):

        self.xvals = xvals
        self.mu    = mu
        self.sigma = sigma

        self._lognormal_pdf()

    def _lognormal_pdf(self):
        """gaussian distribution pdf"""
        self.pdf (1./self.sigma/self.xvals/np.sqrt(2.*np.pi)) * np.exp(-0.5*((np.log(self.xvals)-self.mu)/self.sigma)**2.)



class Maxwellian():
    def __init__(self,xvals,sigma=1.):

        self.xvals = xvals
        self.sigma = sigma

        self._maxwellian_pdf()
        self._maxwellian_cdf()

    def _maxwellian_pdf(self):
        """maxwellian distribution pdf"""
        self.pdf = np.sqrt(2./np.pi)*self.xvals*self.xvals*np.exp(-(self.xvals*self.xvals/2./self.sigma/self.sigma))/self.sigma/self.sigma/self.sigma

    def _maxwellian_cdf(self):
        """maxwellian distribution cdf"""
        self.cdf = erf(self.xvals/np.sqrt(2.)/self.sigma) - np.sqrt(2./np.pi)*self.xvals*np.exp(-(self.xvals*self.xvals/2./self.sigma/self.sigma))/self.sigma


class MultiGaussian():

    def __init__(self,xvals):

        self.xvals = xvals

    def trivariate_gaussian_pdf(self,sigma,cen1=0.,cen2=0.,cen3=0.):
        """trivariate gaussian distribution pdf, with uniform
        widths, but optional different centroids"""
        sigma1 = sigma
        sigma2 = sigma
        sigma3 = sigma
        norm   = 4.*np.pi/(sigma1*sigma2*sigma3)/3.

        pdf = norm*(self.xvals**2.)*np.exp(
            -((self.xvals-cen1)**2./(2.*sigma1**2.))
            -((self.xvals-cen2)**2./(2.*sigma2**2.))
            -((self.xvals-cen3)**2./(2.*sigma3**2.)))

        self.pdf = pdf/np.nansum(pdf*(self.xvals[1]-self.xvals[0]))

        self._make_cdf()

    def _make_cdf(self):
        """make a numerical cdf from a pdf"""
        self.cdf = np.cumsum(self.pdf*(self.xvals[1]-self.xvals[0]))



class Normal():
    """an alias for Gaussian"""
    def __init__(self,xvals,mu=0.,sigma=1.):
       G = Gaussian(xvals,mu,sigma)
       self.xvals = G.xvals
       self.mu    = G.mu
       self.sigma = G.sigma
       self.pdf   = G.pdf
       self.cdf   = G.cdf
