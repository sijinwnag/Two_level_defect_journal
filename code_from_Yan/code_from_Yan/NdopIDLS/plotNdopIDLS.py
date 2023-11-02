import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import matplotlib
from semiconductor.material.thermal_velocity import ThermalVelocity as Vel_th
from semiconductor.material.intrinsic_carrier_density import IntrinsicCarrierDensity as NI
#from semiconductor.electrical.mobility import Mobility as mob
from semiconductor.electrical.ionisation import Ionisation as Ion
from semiconductor.general_functions import carrierfunctions as CF
from scipy.optimize import minimize
import matplotlib.colors as colors
matplotlib.pyplot.switch_backend('Qt5Agg')
matplotlib.rcParams.update({'font.size': 12})

kb = const.k / const.e

def twolevelSRH_full(nxc, Ndop, T, doptype, Nt, Et1, Et2, sigma_e1, sigma_e2, sigma_h1, sigma_h2, **kwarg):
    ni = NI().update(temp=T)
    ni = ni
    ve, vh = Vel_th().update(temp=T)
    ve = ve[0]
    if doptype == 'p':
        Na = Ion(temp=T).update_dopant_ionisation(
            N_dop=Ndop, nxc=0, impurity='boron')
        Na = Na[0]
        Nd = 0
    elif doptype == 'n':
        Nd = Ion(temp=T).update_dopant_ionisation(
            N_dop=Ndop, nxc=0, impurity='phosphorous')
        Nd = Nd[0]
        Na = 0
    n0, p0 = CF.get_carriers(Na=Na, Nd=Nd, nxc=0, temp=T, ni=ni)
    alpha_e1 = sigma_e1 * ve
    alpha_h1 = sigma_h1 * vh
    alpha_e2 = sigma_e2 * ve
    alpha_h2 = sigma_h2 * vh
    n1 = ni * np.exp(Et1 / kb / T)
    p1 = ni * np.exp(-Et1 / kb / T)
    n2 = ni * np.exp(Et2 / kb / T)
    p2 = ni * np.exp(-Et2 / kb / T)
    n = n0 + nxc
    p = p0 + nxc
    R = Nt * (n * p - ni**2) / (1 + ((alpha_e1 * n1 + alpha_h1 * p) / (alpha_e1 * n + alpha_h1 * p1)) + ((alpha_e2 * n + alpha_h2 * p2) / (alpha_e2 *
                                                                                                                                       n2 + alpha_h2 * p))) * ((alpha_e1 * alpha_h1 / (alpha_e1 * n + alpha_h1 * p1)) + (alpha_e2 * alpha_h2 / (alpha_e2 * n2 + alpha_h2 * p)))
    tau = nxc / R
    return tau

Ndop= 1e16

# doptype = 'p'
Nt = 1e12
Et1s = -0.1
Et2s = -0.5
sigma_e1s =2e-13
sigma_h1s = 1e-14

sigma_e2s =4e-14
sigma_h2s = 1e-13

ve300, vh300 = Vel_th().update(temp=300)
ve300 = ve300[0]

taun1s = 1 / Nt / ve300 / sigma_e1s
taun2s = 1 / Nt / ve300 / sigma_e2s

k1s = sigma_e1s / sigma_h1s
k2s = sigma_e2s / sigma_h2s

Ndoplist = [1e16, 1e13, 1e13, 1e16]
typelist = ['n', 'n', 'p', 'p']
T = 300
ni = NI().update(temp=T)
ni = ni[0]
ve, vh = Vel_th().update(temp=T)
ve = ve[0]
nxc = np.logspace(12, 17, 50)

taulist = []
nlist = []
plist = []
nilist = []
velist = []
vhlist = []
for Ndop, doptype in zip(Ndoplist, typelist):
    tausimu = twolevelSRH_full(nxc=nxc, Ndop=Ndop, T=T, doptype=doptype, Nt=Nt, Et1=Et1s,
                               Et2=Et2s, sigma_e1=sigma_e1s, sigma_e2=sigma_e2s, sigma_h1=sigma_h1s, sigma_h2=sigma_h2s)
    taulist.append(tausimu)
    if doptype == 'p':
        Na = Ion(temp=T).update_dopant_ionisation(
            N_dop=Ndop, nxc=0, impurity='boron')
        Na = Na[0]
        Nd = 0
    elif doptype == 'n':
        Nd = Ion(temp=T).update_dopant_ionisation(
            N_dop=Ndop, nxc=0, impurity='phosphorous')
        Nd = Nd[0]
        Na = 0
    n0, p0 = CF.get_carriers(Na=Na, Nd=Nd, nxc=0, temp=T, ni=ni)
    n0 = n0[0]
    p0 = p0[0]
    n = n0 + nxc
    p = p0 + nxc
    nlist.append(n)
    plist.append(p)

Et1list = np.linspace(-0.6, 0.6, 50)
Et2list = np.linspace(-0.6, 0.6, 50)
Etlist = []
for Et1 in Et1list:
    for Et2 in Et2list:
        Etlist.append([Et1, Et2])

taun1listr = np.load('nnNdopIDLS_tau1.npy')
taun2listr = np.load('nnNdopIDLS_tau2.npy')
k1listr = np.load('nnNdopIDLS_k1.npy')
k2listr = np.load('nnNdopIDLS_k2.npy')
residuallistr = np.load('nnNdopIDLS_res.npy')

optind = np.argmin(residuallistr)
Et1opt = Etlist[optind][0]
Et2opt = Etlist[optind][1]
optind = np.unravel_index(optind, (len(Et1list), len(Et2list)))
taun1opt = taun1listr[optind]
taun2opt = taun2listr[optind]
k1opt = k1listr[optind]
k2opt = k2listr[optind]
resiopt = residuallistr[optind]

print(Et1opt, Et2opt, taun1opt, taun2opt, k1opt, resiopt)

plt.figure('fitting')
for taum, Ndop, doptype in zip(taulist, Ndoplist, typelist):

    lm=plt.plot(nxc,taum,'.')
    tauopt = twolevelSRH_full(nxc=nxc, Ndop=Ndop, T=T, doptype=doptype, Nt=Nt, Et1=Et1opt, Et2=Et2opt, sigma_e1=1 / taun1opt / Nt /
                              ve300, sigma_h1=1 / taun1opt / Nt / ve300 / k1opt, sigma_e2=1 / taun2opt / Nt / ve300, sigma_h2=1 / taun2opt / Nt / ve300 / k2opt)
    plt.plot(nxc,tauopt,'--',color=lm[0].get_color())

plt.title('Residual={:.5e}'.format(resiopt))
plt.loglog()
plt.xlabel(r'Excess carrier density [$\rm cm^{-3}$]')
plt.ylabel(r'Lifetime [s]')


extent = (Et2list[0] - (Et2list[1] - Et2list[0]) / 2, Et2list[-1] + (Et2list[-1] - Et2list[-2]) / 2,
          Et1list[0] - (Et1list[1] - Et1list[0]) / 2, Et1list[-1] + (Et1list[-1] - Et1list[-2]) / 2)

plt.figure('Resudual')
im1 = plt.imshow(residuallistr,extent =extent, cmap=plt.cm.inferno, aspect='equal', origin = 'lower',norm=colors.LogNorm())
plt.colorbar(im1)
plt.clim(3e-7, 3)
#plt.plot(Etlist[optind][1],Etlist[optind][0],'bo')
plt.xlabel(r'$E_{t2}-E_{i} \/\/ \rm [eV]$')
plt.ylabel(r'$E_{t1}-E_{i} \/\/ \rm [eV]$')

#
#plt.figure('sn1')
#im1 = plt.imshow(sigma_e1listr,extent =extent, cmap=plt.cm.inferno, aspect='equal', origin = 'lower',norm=colors.LogNorm())
#plt.colorbar(im1)
#plt.plot(Et2s,Et1s,'ro')
#plt.plot(Etlist[optind][1],Etlist[optind][0],'bo')
#
#
#plt.figure('sn2')
#im1 = plt.imshow(sigma_p1listr,extent =extent, cmap=plt.cm.inferno, aspect='equal', origin = 'lower',norm=colors.LogNorm())
#plt.colorbar(im1)
#plt.plot(Et2s,Et1s,'ro')
#plt.plot(Etlist[optind][1],Etlist[optind][0],'bo')
#
#
#plt.figure('sp1')
#im1 = plt.imshow(sigma_e2listr,extent =extent, cmap=plt.cm.inferno, aspect='equal', origin = 'lower',norm=colors.LogNorm())
#plt.colorbar(im1)
#plt.plot(Et2s,Et1s,'ro')
#plt.plot(Etlist[optind][1],Etlist[optind][0],'bo')
#
#
#plt.figure('sp2')
#im1 = plt.imshow(sigma_p2listr,extent =extent, cmap=plt.cm.inferno, aspect='equal', origin = 'lower',norm=colors.LogNorm())
#plt.colorbar(im1)
#plt.plot(Et2s,Et1s,'ro')
#plt.plot(Etlist[optind][1],Etlist[optind][0],'bo')

#
plt.show()
