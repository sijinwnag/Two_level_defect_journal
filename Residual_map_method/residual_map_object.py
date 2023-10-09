# %%-- Imports:
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
import time


class residual_map():
    '''
    Residual map method.
    '''
    
    def __init__(self):
        '''
        Doplist: a list of number for doping values corresponding to hte values in temperature list
        Tlist: a list of integer representing the temperature measured for lifetime.
        Typelist: a list of doping type corresponding to Tlist
        nxcspace: the excess carrier concentration array
        Et1list, Et2list: np array containing hte Et1 and Et2
        Et2: real solution for simulation
        Et1: real solution for simulation of Et1
        se1, sp1, se2, sp2: the real solution for capture cross sections
        Nt: the real solution for defect concentration

        '''
        self.Doplist = [1e15, 1e15, 1e15, 1e15, 1e15, 1e15]
        self.Tlist = [150, 200, 250, 300, 350, 400]
        self.Typelist=['p', 'p', 'p', 'p', 'p', 'p']
        self.nxclist=[]
        self.nxcspace = np.logspace(13,17,50)
        self.taulist=[]

        self.Et1list=np.linspace(-0.55,0.55,56)
        self.Et2list=np.linspace(-0.55,0.55,56)
        #Et1list=[0]
        #Et2list=[0]

        # define the real solution for simulation:
        self.Et2 = -0.3
        self.Et1 = 0.15
        self.se1 = 1e-14
        self.sp1 = self.se1/18
        self.se2 = self.se1/103.2
        self.sp2 = self.se2*86
        self.Nt=1e12

        # define the measured experimental lifetime:
        self.tauexp = np.array

    
    def residual_calculator(self):
        '''
        This function calculate the residuals
        '''
        # load the data from the obeject
        Doplist = self.Doplist
        Tlist = self.Tlist
        Typelist = self.Typelist
        nxclist = self.nxclist
        nxcspace = self.nxcspace
        taulist = self.taulist
        Et1list = self.Et1list
        Et2list = self.Et2list
        Et1 = self.Et1
        Et2 = self.Et2
        se1 = self.se1
        se2 = self.se2
        sp1 = self.sp1
        sp2 = self.sp2
        Nt = self.Nt

        # start recording the time
        start_time = time.time()

        # Define and calcualte constants:
        matplotlib.pyplot.switch_backend('Qt5Agg')
        kb = const.k / const.e

        ve300, vh300 = Vel_th().update(temp=300)
        ve300 = ve300[0]


        # Sah Shockley equation definition:
        def twolevelSRH_full(nxc, Ndop, T, doptype, Nt, Et1, Et2, sigma_e1, sigma_e2, sigma_h1, sigma_h2, **kwarg):
            ni = NI().update(temp=T)
            ni = ni[0]
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


        # for loop one
        for Ndop,T,doptype in zip(Doplist,Tlist,Typelist):
            nxc=nxcspace
            tau = twolevelSRH_full(nxc=nxc,Ndop=Ndop,T=T,doptype=doptype, Nt=Nt, Et1=Et1, Et2=Et2,sigma_e1=se1, sigma_h1=sp1,sigma_e2=se2, sigma_h2=sp2)
            nxclist.append(nxc)
            taulist.append(tau)

        nlist=[]
        plist=[]
        nilist=[]
        velist=[]
        vhlist=[]


        # For loop 2:
        for doptype,nxc,T,Ndop in zip(Typelist,nxclist,Tlist,Doplist):
            ni = NI().update(temp=T)
            ni = ni[0]
            ve, vh = Vel_th().update(temp=T)
            ve = ve[0]
            nilist.append(ni)
            velist.append(ve)
            vhlist.append(vh)
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
            n0=n0[0]
            p0=p0[0]
            n = n0 + nxc
            p = p0 + nxc
            nlist.append(n)
            plist.append(p)


        # for loop 3:
        taun1list=[]
        taun2list=[]
        k1list=[]
        k2list=[]
        residuallist=[]
        Etlist=[]

        Nt=1e12
        count=0
        for Et1 in Et1list:
            for Et2 in Et2list:
                Etlist.append([Et1,Et2])
                def squareresidual(siglist, **kwarg):
                    taun1=siglist[0]*1e-5
                    taun2=siglist[1]*1e-5
                    k1=siglist[2]
                    k2=siglist[3]
                    squareerror = 0
                    for T, nxc, ve, vh, ni, n, p, taum in zip(Tlist, nxclist, velist, vhlist, nilist, nlist, plist, taulist):
                        n1 = ni * np.exp(Et1 / kb / T)
                        p1 = ni * np.exp(-Et1 / kb / T)
                        n2 = ni * np.exp(Et2 / kb / T)
                        p2 = ni * np.exp(-Et2 / kb / T)
                        R = (n * p - ni**2) *((ve*vh/taun1/ve300/(k1*ve*n+vh*p1))+(ve*vh/taun2/ve300/(k2*ve*n2+vh*p)))/(1+((k1*ve*n1+vh*p)/(k1*ve*n+vh*p1))+((k2*ve*n+vh*p2)/(k2*ve*n2+vh*p)))
                        taus = nxc / R
                        squareerror += np.sum((taus-taum)*(taus-taum)/taus/taus)/len(nxc)
                    return squareerror

                res0=100
                for i in range(100):
                    try:
                        fitres = minimize(squareresidual, x0=[10**(np.random.sample()*10-5),10**(np.random.sample()*10-5),10**(np.random.sample()*10-5),10**(np.random.sample()*10-5)],bounds=[(1e-5,1e5),(1e-5,1e5),(1e-5,1e5),(1e-5,1e5)],method='L-BFGS-B', options={'eps':1e-9,'gtol':1e-12})
                        if fitres.fun < res0:
                            res0 = fitres.fun
                            truefitres = fitres
                    except RuntimeError:
                        pass
                try:
                    taun1list.append(truefitres.x[0]*1e-5)
                    taun2list.append(truefitres.x[1]*1e-5)
                    k1list.append(truefitres.x[2])
                    k2list.append(truefitres.x[3])
                    residuallist.append(truefitres.fun)
                    del truefitres
                except NameError:
                    taun1list.append(0)
                    taun2list.append(0)
                    k1list.append(0)
                    k2list.append(0)
                    residuallist.append(100)
                count += 1
                print(str(count)+'/'+str(len(Et1list)*len(Et2list))+' finished')


        # paramatarisation
        taun1list=np.asarray(taun1list)
        taun2list=np.asarray(taun2list)
        k1list=np.asarray(k1list)
        k2list=np.asarray(k2list)
        residuallist=np.asarray(residuallist)

        optind = np.argmin(residuallist)
        Et1opt= Etlist[optind][0]
        Et2opt=Etlist[optind][1]
        taun1opt= taun1list[optind]
        taun2opt= taun2list[optind]
        k1opt= k1list[optind]
        k2opt= k2list[optind]
        resiopt=residuallist[optind]

        # dataprocess 2
        taun1listr = np.reshape(taun1list,(len(Et1list),len(Et2list)))
        taun2listr = np.reshape(taun2list,(len(Et1list),len(Et2list)))
        k1listr = np.reshape(k1list,(len(Et1list),len(Et2list)))
        k2listr = np.reshape(k2list,(len(Et1list),len(Et2list)))
        residuallistr = np.reshape(residuallist,(len(Et1list),len(Et2list)))
        extent = (Et2list[0], Et2list[-1], Et1list[0], Et1list[-1])
        optind = np.argmin(residuallist)

        # end recording the time
        end_time = time.time()

        dt = end_time - start_time
        print(f'Time taken: {dt} seconds')

        # store the residual map solution output into the object
        self.Et1opt = Et1opt # optimal Et1 to minimize the residual
        self.Et2opt = Et2opt # optimal Et2 to minimize the residual
        self.taun1opt = taun1opt
        self.taun2opt = taun2opt
        self.k1opt = k1opt
        self.k2opt = k2opt
        self.resiopt = resiopt
        print('optimal Et1 is ' + str(Et1opt))
        print('optimal Et2 is ' + str(Et2opt))
        print('optimal k1 is ' + str(k1opt))
        print('optimal k2 is ' + str(k2opt))
        # store the arrays for plotting into the object
        self.taun1listr = taun1listr
        self.taun2listr = taun2listr
        self.k1listr = k1listr
        self.k2listr = k2listr
        self.residuallistr = residuallistr 
        self.extent = extent


    def data_savor(self):
        '''
        Save the data of residual map as npy files
        '''
        np.save('taun1list_diff_d.npy',self.taun1list)
        np.save('taun2list_diff_d.npy',self.taun2list)
        np.save('taun1listr_diff_d.npy',self.taun1listr)
        np.save('taun2listr_diff_d.npy',self.taun2listr)
        np.save('k1list_diff_d.npy',self.k1list)
        np.save('k2list_diff_d.npy',self.k2list)
        np.save('k1listr_diff_d.npy',self.k1listr)
        np.save('k2listr_diff_d.npy',self.k2listr)
        np.save('residuallist_diff_d.npy',self.residuallist)
        np.save('residuallistr_diff_d.npy',self.residuallistr)
        np.save('Etlist_diff_d.npy',self.Etlist)


    def data_loader(self):
        '''
        Load the saved data
        '''
        self.taun1list = np.load(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Yan_DPSS_code\taun1list.npy')
        self.Etlist = np.load(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Yan_DPSS_code\Etlist.npy')
        self.Et1list = self.Etlist[:, 0]
        self.Et2list = self.Etlist[:, 1]
        self.taun2list = np.load(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Yan_DPSS_code\taun2list.npy')
        self.k1list = np.load(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Yan_DPSS_code\k1list.npy')
        self.k1listr = np.load(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Yan_DPSS_code\k1listr.npy')
        self.k2list = np.load(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Yan_DPSS_code\k2list.npy')
        self.k2listr = np.load(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Yan_DPSS_code\k2listr.npy')
        self.residuallist = np.load(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Yan_DPSS_code\residuallist.npy')
        self.residuallistr = np.load(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Yan_DPSS_code\residuallistr.npy')
        self.taun1listr = np.load(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Yan_DPSS_code\taun1listr.npy')
        self.taun2list = np.load(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Yan_DPSS_code\taun2list.npy')
        self.taun2listr = np.load(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Yan_DPSS_code\taun2listr.npy')
        self.extent = (self.Et2list[0], self.Et2list[-1], self.Et1list[0], self.Et1list[-1])
        self.optind = np.argmin(self.residuallist)