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

        self.Et1list=np.linspace(0,0.5,5)
        self.Et2list=np.linspace(-0.5,0,5)
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

        # define whether simulate or use experimental data.
        self.simulate=False

        # define the measured experimental lifetime:
        # p type 303K 8.5e14 doping density.
        self.dn_p_303_8_5e14 = np.array([1e13,
                    27534022784368,
                    31550792957346.9,
                    36153545162406.1,
                    41427764734064.4,
                    47471407939424.3,
                    54396721286250.3,
                    62332326238771,
                    71425608059923.3,
                    81845453147175,
                    93785385701533.1,
                    107467161986003,
                    123144888928437,
                    141109743562151,
                    161695380956878,
                    185284131079681,
                    212314099678011,
                    243287305067151,
                    278779001943822,
                    319448365393920,
                    366050733524800,
                    419451636100292,
                    480642869727488,
                    550760918154195,
                    631108059790913,
                    723176554480104,
                    828676358725287,
                    949566884125385,
                    1088093388852790,
                    1246828678061630,
                    1428720887713460,
                    1637148239293100,
                    1875981782354990,
                    2149657290195900,
                    2463257643947610,
                    2822607235180910
                    ])
        self.tau_p_303_8_5e14 = np.array([0.0004,
                            0.000601262740429958,
                            0.000658665531867146,
                            0.000718181702553165,
                            0.000776728809090367,
                            0.000827604665586884,
                            0.000889510134291615,
                            0.000950120788351192,
                            0.000983243203324339,
                            0.000980289928468689,
                            0.00109167977048749,
                            0.00123483478114777,
                            0.00133801501230349,
                            0.00142922981842247,
                            0.00153012157703939,
                            0.00171956018953192,
                            0.00173043052728484,
                            0.00188110130147385,
                            0.0019827315220447,
                            0.00204251291116385,
                            0.002171575705917,
                            0.00234573842998269,
                            0.00243317703947774,
                            0.00245459042800956,
                            0.00251241177161198,
                            0.00266486217885697,
                            0.00266192674623504,
                            0.00272059528039267,
                            0.00288529927412184,
                            0.00293309196337702,
                            0.00289984139428033,
                            0.0029691330081956,
                            0.00300099571696042,
                            0.00296425092032784,
                            0.00300583674711878,
                            0.00299323759568535,
                            ])
        # p type 343K 2e15 doping density.
        self.dn_p_343_2e15 = np.array([103187160115410,
                        112716917530961,
                        123126787126145,
                        134498050870162,
                        146919497455412,
                        160488115574167,
                        175309851222190,
                        191500434942445,
                        209186285468160,
                        228505496821028,
                        249608916571983,
                        272661323683771,
                        297842715132119,
                        325349711351646,
                        355397091480464,
                        388219470390916,
                        424073130600979,
                        463238023370204,
                        506019954605061,
                        552752972641620,
                        603801976549742,
                        659565565324876,
                        720479150214384,
                        787018354479978,
                        859702727142161,
                        939099799704190,
                        1025829517531140,
                        1120569080485000,
                        1224058230612260,
                        1337105028171170,
                        1460592161098680,
                        1595483837182800,
                        1742833312754940,
                        1903791116688320,
                        2079614033915900,
                        2271674918613470,
                        2481473413670080,
                        2710647660145780,
                        2960987088145640,
                        3234446388983540,
                        3533160777732550,
                        3859462665334400,
                        4215899870446730,
                        4605255513229000,
                        5030569746401210,
                        5495163493254960,
                        6002664377967210,
                        6557035050684830,
                        7162604128546910,
                        7824099994237340,
                        8546687715972350,
                        9336009377200320,
                        10198228130914700,
                        11140076322563500,
                        12168908057306600,
                        13292756622075600,
                        14520397210794800,
                        15861415442535600,
                        17326282207602000,
                        18926435425959500,
                        20674369356390000,
                        22583732153714200,
                        24669432435822800,
                        ])
        self.tau_p_343_2e15 = np.array([0.000348695673907163,
                        0.000279921259379822,
                        0.000313862430652565,
                        0.000255042067306108,
                        0.000281305940339077,
                        0.00033007376479769,
                        0.000297656585383869,
                        0.000300313968836968,
                        0.000262957265597958,
                        0.000274404603619128,
                        0.000299781708818235,
                        0.000288108710028952,
                        0.000264378878545744,
                        0.000268657042758276,
                        0.000266069901834315,
                        0.000250112044120745,
                        0.000262857274637482,
                        0.000258346763800657,
                        0.000247299168078695,
                        0.000244965508926453,
                        0.000238261912360439,
                        0.000237764258977961,
                        0.000232446557912611,
                        0.000224376428250502,
                        0.000216949451532133,
                        0.000219058490523047,
                        0.000215827138330216,
                        0.000211994738667066,
                        0.000207994854377211,
                        0.000208634885881568,
                        0.000202198629604142,
                        0.000200437506753696,
                        0.00019718472744064,
                        0.000196233146769916,
                        0.000191958298465989,
                        0.000189884088006154,
                        0.000190179253627594,
                        0.000187644219733863,
                        0.00018621695005858,
                        0.000184399107272355,
                        0.000183754410809022,
                        0.000181935706926214,
                        0.000180363678878678,
                        0.000179180695273686,
                        0.000178375080510873,
                        0.000177347696132585,
                        0.000176940086271837,
                        0.000176442278268817,
                        0.000176488737810985,
                        0.000176616414287653,
                        0.000176724299175871,
                        0.000176387149168174,
                        0.000176147372953221,
                        0.000175968818192952,
                        0.000176617038046456,
                        0.000176817267808765,
                        0.000177253732061947,
                        0.000177852249515638,
                        0.000178411485205063,
                        0.000179228578622751,
                        0.000180091525205379,
                        0.000181200559918785,
                        0.000180869344610297,
                        ])
        # n type 303K 2e15 doping density.
        self.dn_n_303_2e15 = np.array([117542045198389,
                        128978238867442,
                        141527111198971,
                        155296919699079,
                        170406454733019,
                        186986064314375,
                        205178778600603,
                        225141543795779,
                        247046576105301,
                        271082847422922,
                        297457715566948,
                        326398713129446,
                        358155510370618,
                        393002069091997,
                        431239006069643,
                        473196186436365,
                        519235569385757,
                        569754330747564,
                        625188289372441,
                        686015666885052,
                        752761213240487,
                        826000733674551,
                        906366056101416,
                        994550481811691,
                        1091314766493410,
                        1197493683173400,
                        1314003223696740,
                        1441848500870470,
                        1582132419442390,
                        1736065190718320,
                        1904974772899290,
                        2090318327205880,
                        2293694788620600,
                        2516858659694030,
                        2761735146412570,
                        3030436766702530,
                        3325281574850400,
                        3648813159058730,
                        4003822584654080,
                        4393372472248330,
                        4820823418572380,
                        5289862987911570,
                        5804537524247930,
                        6369287058548970,
                        6988983612343110,
                        7668973228022220,
                        8415122088461170,
                        9233867123822200,
                        10132271542122900,
                        11118085762622200,
                        12199814277690000,
                        13386789019967500,
                        14689249867749300,
                        16118432983094500,
                        17686667744751000,
                        19407483112118200,
                        21295724337838300,
                        23367681035879500,
                        25641227709939200,
                        ])
        self.tau_n_303_2e15 = np.array([0.000149384864464721,
                        0.000158357872845068,
                        0.000149607268649753,
                        0.000150241353019561,
                        0.000158521950205039,
                        0.000149555492483859,
                        0.000157466747952028,
                        0.000155645975644549,
                        0.000153205176128088,
                        0.000154448216428553,
                        0.000152528465847741,
                        0.000147887629728663,
                        0.000144695529021539,
                        0.000145376340809639,
                        0.000149742703021257,
                        0.000147641350414667,
                        0.000141002343244598,
                        0.000140083943671049,
                        0.000140936981403673,
                        0.000139967833982669,
                        0.000138603822287922,
                        0.000136697895298875,
                        0.000137674547424659,
                        0.000135595150661463,
                        0.00013482495406114,
                        0.000134916088640091,
                        0.000134127594220305,
                        0.00013450507766738,
                        0.000134763608200232,
                        0.000134529204678436,
                        0.000133958963800213,
                        0.000134693006295653,
                        0.000134552692201467,
                        0.000134647330875716,
                        0.000134933255807573,
                        0.000135549989280338,
                        0.00013624189030023,
                        0.00013623431408167,
                        0.000136358353839015,
                        0.000136394967887297,
                        0.000137606800600332,
                        0.000137845260831909,
                        0.000138498047210152,
                        0.000139283596650858,
                        0.000140305444927538,
                        0.00014142582201297,
                        0.000142192773028039,
                        0.000143099669323422,
                        0.000144138104499054,
                        0.000145153177374482,
                        0.000146098255761071,
                        0.000147599643835638,
                        0.000148613356920093,
                        0.000149818436393802,
                        0.000151109143237704,
                        0.000152411563988999,
                        0.000153877233412181,
                        0.000155072782534966,
                        0.00015167544798772,
                        ])

    # Sah Shockley equation definition:
    def twolevelSRH_full(nxc, Ndop, T, doptype, Nt, Et1, Et2, sigma_e1, sigma_e2, sigma_h1, sigma_h2, **kwarg):
        kb = const.k / const.e
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
    

    def residual_calculator(self):
        '''
        This function calculate the residuals
        '''

        # update the doping and Temperature lits if using experimental data:
        if not self.simulate:
            print('using experimental data')
            self.Doplist = [8.5e14, 2e15]
            self.Tlist = [303, 303]
            self.Typelist=['p', 'n']
            self.nxclist=[]
            self.nxcspace = np.logspace(13,17,50)
            self.taulist=[]

        
        # load the data from the object
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
        # store into the object
        self.ve300 = ve300
        self.vh300 = vh300
        ve300 = ve300[0]

        # for loop one
        for Ndop, T, doptype in zip(Doplist, Tlist, Typelist):
            if self.simulate:
                nxc=nxcspace
                tau = self.twolevelSRH_full(nxc=nxc,Ndop=Ndop,T=T,doptype=doptype, Nt=Nt, Et1=Et1, Et2=Et2,sigma_e1=se1, sigma_h1=sp1,sigma_e2=se2, sigma_h2=sp2)
                nxclist.append(nxc)
                taulist.append(tau)
            else:
                if doptype == 'p':
                    if T == 303 and Ndop == 8.5e14:
                        nxc = self.dn_p_303_8_5e14
                        tau = self.tau_p_303_8_5e14
                        Nt = 1e12
                elif doptype == 'n':
                    if T == 303 and Ndop == 2e15:
                        nxc = self.dn_n_303_2e15
                        tau = self.tau_n_303_2e15
                        Nt = 1e12
                    elif T == 343 and Ndop == 2e15:
                        nxc = self.dn_p_343_2e15
                        tau = self.tau_p_343_2e15
                        Nt = 1e12
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
                        # taus = self.twolevelSRH_full(nxc=nxc,Ndop=Ndop,T=T,doptype=doptype, Nt=Nt, Et1=Et1, Et2=Et2,sigma_e1=taun1, sigma_h1=taun1,sigma_e2=taun2, sigma_h2=taun2)
                        n1 = ni * np.exp(Et1 / kb / T)
                        p1 = ni * np.exp(-Et1 / kb / T)
                        n2 = ni * np.exp(Et2 / kb / T)
                        p2 = ni * np.exp(-Et2 / kb / T)
                        R = Nt * (n * p - ni**2) *((ve*vh/taun1/ve300/(k1*ve*n+vh*p1))+(ve*vh/taun2/ve300/(k2*ve*n2+vh*p)))/(1+((k1*ve*n1+vh*p)/(k1*ve*n+vh*p1))+((k2*ve*n+vh*p2)/(k2*ve*n2+vh*p)))
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


    def fit_plotter(self):
        # Plot the fitting curve
        plt.figure(num='fitting', facecolor='white')

        for taum, T, Ndop, doptype, nxc in zip(self.taulist, self.Tlist, self.Doplist, self.Typelist, self.nxclist):
            lm=plt.plot(nxc,taum,'.')
            tauopt=self.twolevelSRH_full(nxc=nxc,Ndop=Ndop, T=T, doptype=doptype, Nt=self.Nt, Et1=self.Et1opt, Et2=self.Et2opt,sigma_e1=1/self.taun1opt/self.Nt/self.ve300, sigma_h1=1/self.taun1opt/self.Nt/self.ve300/self.k1opt,sigma_e2=1/self.taun2opt/self.Nt/self.ve300, sigma_h2=1/self.taun2opt/self.Nt/self.ve300/self.k2opt)
            plt.plot(nxc,tauopt,'--',color=lm[0].get_color())

        plt.title('Residual={:.5e}'.format(self.resiopt))
        plt.loglog()
        plt.xlabel(r'Excess carrier density [$\rm cm^{-3}$]')
        plt.ylabel(r'Lifetime [s]')
        plt.savefig('fitting curve' + '.png')
        plt.show()


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