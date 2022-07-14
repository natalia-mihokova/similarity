"""
this file tests the analytic distributions, with defaults
Writting something to test 
"""
import numpy as np
import pkg_resources
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# bring in some different analytical distributions
from similarity import analytical as an

xvals = np.linspace(-10.,10.,1000)

Cauchy   = an.Cauchy(xvals)
Logistic = an.Logistic(xvals)
Normal   = an.Normal(xvals)
Chi2     = an.Chi2(xvals,k=3.)
Lognormal= an.Lognormal(xvals)
Maxwell  = an.Maxwellian(xvals)

plt.figure(figsize=(4,3))

plt.plot(Normal.xvals,Normal.pdf      ,color=cm.viridis(0./5.,1.),linestyle=(0, (3, 1, 1, 1)),label='normal',lw=1.)
plt.plot(Logistic.xvals,Logistic.pdf  ,color=cm.viridis(1./5.,1.),linestyle=(0, (3, 2, 1, 2)),label='logistic',lw=1.)
plt.plot(Cauchy.xvals,Cauchy.pdf      ,color=cm.viridis(2./5.,1.),linestyle=(0, (3, 3, 1, 3)),label='cauchy',lw=1.)
plt.plot(Chi2.xvals,Chi2.pdf          ,color=cm.viridis(3./5.,1.),linestyle=(0, (3, 4, 1, 4)),label='chi2',lw=1.)
plt.plot(Lognormal.xvals,Lognormal.pdf,color=cm.viridis(4./5.,1.),linestyle=(0, (3, 5, 1, 5)),label='lognormal',lw=1.)
plt.plot(Maxwell.xvals,Maxwell.pdf    ,color=cm.viridis(5./5.,1.),linestyle=(0, (3, 6, 1, 6)),label='maxwellian',lw=1.)

plt.xlabel('sample points')
plt.ylabel('pdf')
plt.legend()

plt.tight_layout()
plt.savefig('testpdfs.png')

plt.figure(figsize=(4,3))

plt.plot(Normal.xvals,Normal.cdf      ,color=cm.viridis(0./5.,1.),linestyle=(0, (3, 1, 1, 1)),label='normal',lw=1.)
plt.plot(Logistic.xvals,Logistic.cdf  ,color=cm.viridis(1./5.,1.),linestyle=(0, (3, 2, 1, 2)),label='logistic',lw=1.)
plt.plot(Cauchy.xvals,Cauchy.cdf      ,color=cm.viridis(2./5.,1.),linestyle=(0, (3, 3, 1, 3)),label='cauchy',lw=1.)
plt.plot(Chi2.xvals,Chi2.cdf          ,color=cm.viridis(3./5.,1.),linestyle=(0, (3, 4, 1, 4)),label='chi2',lw=1.)
plt.plot(Lognormal.xvals,Lognormal.cdf,color=cm.viridis(4./5.,1.),linestyle=(0, (3, 5, 1, 5)),label='lognormal',lw=1.)
plt.plot(Maxwell.xvals,Maxwell.cdf    ,color=cm.viridis(5./5.,1.),linestyle=(0, (3, 6, 1, 6)),label='maxwellian',lw=1.)

plt.xlabel('sample points')
plt.ylabel('cdf')
plt.legend()

plt.tight_layout()
plt.savefig('testcdfs.png')
