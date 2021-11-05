# similarity

### A python package for computing similarity scores between different one-dimensional distributions.

-------

Install by running `python setup.py install` in the similarity directory once unpacked.

-------

#### The theory.

Some example data ships with the package, drawn from simulations presented in Donaldson et al. (2022).

The cumulative distribution functions (CDFs) presented in this work exhibit many similarities. We therefore wish to test the null hypothesis of two CDFs being drawn from the same parent distribution in order to quantify when the distributions are so dissimilar as to {\it not} be drawn from the same distribution. We use a lightly modified Kolmogorov-Smirnov (KS) statistic procedure. As the KS statistic may be difficult to interpret, we detail our procedure and calibrate the values of the KS statistic for our study in this Appendix.

Given two empirical CDFs, $C_1$ and $C_2$, sampled at the same $v$ points, we define the KS statistic as $${\mathcal D} = \max_{1 \leq v \leq N}\left(C_1(v) - C_2(v-1),C_2(v) - C_1(v)\right).$$ As $C_1$ and $C_2$ only have values between 0 and 1, the test statistic has the same range: ${\mathcal D}=0$ corresponds to perfectly matched distributions, and ${\mathcal D}=1$ corresponds to distributions that have exactly no overlap. For traditional KS tests, the ${\mathcal D}$ is compared to the CDF of a normal distribution, and may be straightforwardly translated to a probability given a $\chi^2$ distribution. For our application, we require an empirical calibration of ${\mathcal D}$, which we construct by subsampling the velocity CDFs.

In order to compute the KS statistic, $C_1$ and $C_2$ must be sampled at the same points. To compute matched sets from two unmatched CDFs, $c_1$ and $c_2$ (with number of samples $n_{c_1}$ and $n_{c_2}$ respectively), we proceed as follows. Define a new lower bound $c_{\rm max} = \max\left(\min(c_1),\min(c_2)\right)$, upper bound $c_{\rm min} = \min\left(\max(c_1),\max(c_2)\right)$, and sample resolution $\Delta v=(c_{\rm max}-c_{\rm min})/N$ (for a given choice of $N$), to construct a new set of sample points $V = \left(c_{\rm min},c_{\rm min}+\Delta v, c_{\rm min},c_{\rm min}+2\Delta v,\ldots,c_{\rm max}\right)$. For best results, $n_{c_1}$ and $n_{c_2}$ should have similar numbers of samples, and the choice of $N$ should be similar. However, we find in practice that the choice of sample points varied by a factor of 2 does not strongly influence the results. We empirically estimate the uncertainty from choice of $N$ to be $\Delta {\mathcal D}<0.01$.

To calibrate ${\mathcal D}$, we begin with a 'true' empirical distribution $c_1$ and construct $c_2$ by drawing $n_{c_2}={\rm int}(\alpha n_{c_1})$ where $0<\alpha<1$ random samples from $c_1$. By scanning through $\alpha$ values, we can determine the value of $\alpha$ above which results are converged, and the uncertainty in ${\mathcal D}$. In practice, we are probing two different regimes when calculating ${\mathcal D}$: one where $N\approx500$, and one where $N\approx10000$. We report calibrations for both. For the $N\approx500$ case, we find that ${\mathcal D}$ is converged for $\alpha>0.6$, with $\langle{\mathcal D}\rangle = 0.041\pm0.013$. For the $N\approx10000$ case, we find that ${\mathcal D}$ is also converged for $\alpha>0.6$, with $\langle{\mathcal D}\rangle = 0.010\pm0.003$.

For applications in this work, we theorise that the mean values of two CDFs $c_1$ and $c_2$ may be offset relative to one another. To test the dissimilarity of distributions independent of fixed mean offsets, we compute ${\mathcal D}$ for different offset values $\Delta x$, given two distributions $c_1$ and $c_2+\Delta x$. We test values of $\Delta x$ in a range defined by $\pm2(\langle c_1\rangle-\langle c_2\rangle)$. To calibrate uncertainty, we compute ${\mathcal D}$ for different values of $\Delta x$ using the same CDF for $c_1$ and $c_2$. We find the theoretically-expected relationship ${\mathcal D}\approx |\Delta x|/\langle c_1\rangle$ holds, such that the uncertainty in ${\mathcal D}$ may be approximated as ${\mathcal D}\approx |\langle c_1\rangle-\langle c_2\rangle|/\langle c_1\rangle$. Such a search for a minimum value of ${\mathcal D}$ will maximise the chances of finding that the two CDFs are drawn from the same parent distribution, and give the strongest constraints.

One additional shortcoming of the KS statistic is the insensitivity to subtle variations in the tails of the distribution. However, we can place a theoretical limit on the differences by noting that if one were to truncate the distribution $c_2$ at some maximum $v$, then ${\mathcal D}\approx 1-c_1(v)$. That is, if one defined $c_2$ to be the minimum 90\% of the values of $c_1$, $c_1(v)=0.9$, and we would compute ${\mathcal D}\approx0.1$. We may therefore place meaningful upper limits on dissimilarities in the tails of distributions by computing ${\mathcal D}$.

Given the calibrations, we can make statements about the dissimilarity of two distributions given some simple characteristics. Collecting all of the calibration uncertainties, we find that the uncertainty on the minimum value of ${\mathcal D}$ will be primarily set by the number of samples $n_{c_1}$ and $n_{c_2}$, with other uncertainties strongly subdominant except for at very large values of $n_{c_1}$ and $n_{c_2}$. Defining $n\equiv n_{c_1}\approx n_{c_2}$, for two CDFs $c_1$ and $c_2$, we are unable to distinguish between the two CDFs at the 95\% confidence level if ${\mathcal D}<0.016\sqrt{10000/n}.$


### License
-------

This project is Copyright (c) Michael Petersen and licensed under the terms of the two-clause BSD license. See the ``licenses`` folder for more information.
