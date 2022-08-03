"""
class structure for handling different distributions from sets of (x,y) points, with optional weights

"""

import numpy as np

from scipy.interpolate import interp1d



class Distribution():
    """
    class to handle empirical distributions, making pdfs and cdfs from
    individual observations
    """

    def __init__(self,xvalues,weights=1,xbins=-1):
        """
        create empirical distributions from observations

        inputs
        ---------
        xvalues : (array of floats) the x-positions of the observations
        yvalues : (array of floats) the y-values of the observations
        weights : (array of floats, optional) the weights of the observations
        xbins   : (array of floats, optional) the points to bin to

        """

        self.xvalues = xvalues

        if isinstance(weights,int):
            self.weights = np.ones(xvalues.size)
        else:
            # check if the length matches xvalues?
            self.weights = weights

        # if xbins is > 0, use as the binning range. otherwise, make
        # an adaptive binning range
        if isinstance(xbins,int):
            if xbins<0:
                self.xbins = np.linspace(np.nanmin(xvalues),np.nanmax(xvalues),int(np.sqrt(xvalues.size)))
            else:
                self.xbins = np.linspace(np.nanmin(xvalues),np.nanmax(xvalues),xbins)
        else:
            self.xbins = xbins

        self.dx = self.xbins[1]-self.xbins[0]

        # separate binning for the cumulative distributions
        self.xcbins = np.linspace(np.nanmin(xvalues),np.nanmax(xvalues),xvalues.size)
        self.dcx = self.xcbins[1]-self.xcbins[0]

        # make the cdf and pdf
        self._weighted_pdf()
        self._weighted_cdf()

        # set up for homogenisation with another distribution
        self.hxbins = None
        self.hpdf   = None
        self.hcdf   = None


    def _weighted_pdf(self):
        """
        make a very quick 1d pdf, with weights
        """
        vals = np.zeros_like(self.xbins)

        for b in range(0,self.xbins.size):
            if b==self.xbins.size-1:
                w = np.where( (self.xvalues>self.xbins[b]) & (self.xvalues<=self.xbins[b]+self.dx))[0]
            else:
                w = np.where( (self.xvalues>self.xbins[b]) & (self.xvalues<=self.xbins[b+1]))[0]
            vals[b] = np.nansum(self.weights[w])

        # normalise the area to one
        norm = np.nansum(self.dx*vals)

        self.pdf = vals/norm


    def _weighted_cdf(self):
        """
        make a very quick 1d cdf, with weights
        """
        vals = np.zeros_like(self.xcbins)
        for b in range(0,self.xcbins.size):
            if b==self.xcbins.size-1:
                w = np.where((self.xvalues<=self.xcbins[b]+self.dcx))[0]
            else:
                w = np.where((self.xvalues<=self.xcbins[b+1]))[0]
            vals[b] = np.nansum(self.weights[w])

        # normalise the area to one
        norm = np.nanmax(vals)

        self.cdf = vals/norm


    def _un_weighted_cdf(self):
        """make an unweighted cdf, with full resolution"""
        self.vsort = self.yvalues[self.yvalues.argsort()]
        self.wsort = np.cumsum(self.weights[self.yvalues.argsort()])/np.nansum(self.weights)


    def _interpolate_to_grid(self,grid):
        """put distributions on a specified grid"""

        self.hxbins = grid
        self.hpdf   = interp1d(self.xbins, self.pdf, kind='linear')(self.hxbins)
        self.hcdf   = interp1d(self.xcbins, self.cdf, kind='linear')(self.hxbins)



def match_distributions(Distribution1,Distribution2,spacing=-1.,verbose=0):
    """create a uniform grid

    spacing sets the km/s sampling. 1 km/s is a good default.
    """

    if spacing < 0:
        spacing = np.nanmax([Distribution1.dcx,Distribution2.dcx])
        if verbose: print('similarity.empirical.match_distributions: spacing=',spacing)

    # check for overlaps: if none, we can't use this!
    if (np.nanmax(Distribution1.xbins) < np.nanmin(Distribution2.xbins)) | (np.nanmax(Distribution2.xbins) < np.nanmin(Distribution1.xbins)):
        print('similarity.empirical.match_distributions: there is no overlap in these samples!')


    xnew = np.arange(np.nanmax([np.nanmin(Distribution1.xbins),np.nanmin(Distribution2.xbins)]),
                     np.nanmin([np.nanmax(Distribution1.xbins),np.nanmax(Distribution2.xbins)]),
                     spacing)

    Distribution1._interpolate_to_grid(xnew)
    Distribution2._interpolate_to_grid(xnew)
