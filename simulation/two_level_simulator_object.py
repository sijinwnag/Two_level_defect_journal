# %%-- Imports
import sys
# sys.path.append(r'C:\Users\budac\Documents\GitHub\Yoann_code\DPML')
# sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\Yoann_code\DPML')
# from Si import *

# use this line for dell laptop:
# sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\yoann_code_new')
# use this line for workstation:
# sys.path.append(r'C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\Two_level_defect_journal\simulation\DPML')
# sys.path.append(r'C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\Two_level_defect_journal\simulation')
from DPML import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import smtplib
from email.message import EmailMessage
import os
from scipy.signal import savgol_filter

# %%


class two_level_lifetime_generator():
    '''
    This is an object that generate lifetime for two-level-defect based on Sah Shockley Equation
    '''

    def __init__(self):
        '''
        Inputs: 
        SAVEDIR: a string, showing the directory to save the generated data
        FILEPATH: something you don't need to worry or change, just keept it there
        TEMPERATURE: a list of temperature to simulate lifetime.
        DOPING: a list of doping range to simulate lifetime. The length of TEMPERATURE and DOPING should match up, how it will work is that the lifetime will use first temperature with first doping and second temperature with the second doping and so on.
        WAFERTYPE: a string be either'n' or 'p'
        NAME: a string of the file name
        PARAMETERS: a dictionary of parameters regarding the simulation. 
        PARAM: a dictionary of parameters regarding the simulation.

        This function initialize all the parameters for simulation
        '''
        # self.SAVEDIR = r"C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\Two_level_defect_journal\simulation\simulated_data" # you can change this to your own path
        # self.SAVEDIR = r"C:\Users\sijin wang\Documents\GitHub\Two_level_defect_journal\simulation\simulated_data"
        # self.FILEPATH = "advanced_example\\data\\sample_original_L.csv"
        self.TEMPERATURE = [150, 200, 250, 300, 350, 400] # below 400K
        # DOPING = [5e13, 1e14, 5e14, 1e15, 5e15, 1e16] # *len(TEMPERATURE) # make sure T and doping have same length
        # DOPING = [1e15]
        self.DOPING = [1e15] *len(self.TEMPERATURE) # make sure T and doping have same length.
        # if using different doping levels: each temperature will match each doping.
        # so each element of temperature will repeat for length of itself times
        # TEMPERATURE
        self.WAFERTYPE = 'p'
        self.NAME = 'advanced example - multi_level_L'

        self.PARAMETERS = {
            'name': self.NAME,
            'save': False,   # True to save a copy of the printed log, the outputed model and data
            'logML':False,   #   Log the output of the console to a text file
            'n_defects': 8000, # Size of simulated defect data set for machine learning
            'dn_range' : np.logspace(13,17,100),# Number of points to interpolate the curves on
            'non-feature_col':['Mode','Label',"Name","Et_eV_1","Sn_cm2_1","Sp_cm2_1",'k_1','logSn_1','logSp_1','logk_1','bandgap_1',"Et_eV_2","Sn_cm2_2","Sp_cm2_2",'k_2','logSn_2','logSp_2','logk_2','bandgap_2']
        }
        self.PARAM={
                'type': self.WAFERTYPE,                #   Wafer doping type
                'Et_min_1':-0.55,             #   Minimum defect energy level
                'Et_max_1':0.55,              #   Maximum defect energy level
                'Et_min_2':-0.55,             #   Minimum defect energy level
                'Et_max_2':0.55,              #   Maximum defect energy level
                'S_min_1_p':10**(-17),              #   Minimum capture cross section for hole.
                'S_min_1_n':10**(-17),          #   Minimum capture cross section for electron.
                'S_max_1_p':10**(-13),              #   Maximum capture cross section for hole.
                'S_max_1_n':10**(-13),              # maximum capcture cross section for electron.
                'S_min_2_p':1E-17,              #   Minimum capture cross section for hole.
                'S_min_2_n':1E-17,          #   Minimum capture cross section for electron.
                'S_max_2_p':1E-13,              #   Maximum capture cross section for hole.
                'S_max_2_n':1E-13,              # maximum capcture cross section for electron.
                # the parameter to simulate BO:
                # 'Et_min_1':0.15,             #   Minimum defect energy level
                # 'Et_max_1':0.15,              #   Maximum defect energy level
                # 'Et_min_2':-0.3,             #   Minimum defect energy level
                # 'Et_max_2':-0.3,              #   Maximum defect energy level
                # 'S_min_1_p':1e-14/18,              #   Minimum capture cross section for hole.
                # 'S_min_1_n':1e-14,          #   Minimum capture cross section for electron.
                # 'S_max_1_p':1e-14/18,              #   Maximum capture cross section for hole.
                # 'S_max_1_n':1e-14,              # maximum capcture cross section for electron.
                # 'S_min_2_p':1e-14/103.2*86,              #   Minimum capture cross section for hole.
                # 'S_min_2_n':1e-14/103.2,          #   Minimum capture cross section for electron.
                # 'S_max_2_p':1e-14/103.2*86,              #   Maximum capture cross section for hole.
                # 'S_max_2_n':1E-14/103.2,              # maximum capcture cross section for electron.
                'Nt':1E12,                  #   Defect density
                'check_auger':False,     #   Check wether to resample if lifetime is auger-limited
                'noise':'',             #   Enable noiseparam
                'noiseparam':0,         #   Adds noise proportional to the log of Delta n
        }

        # input the experimental data: the lifetime are in seconds
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
  

    def mixture_simulate(self, emailme = False):
        '''
        This function simulate the lifetime for a mixture of one-level and two-level defects.
        '''
        #  Define experiment: every time remember to run this line to refresh the code.
        self.exp = Experiment(SaveDir=self.SAVEDIR, Parameters=self.PARAMETERS)

        # perform the simulation
        db_multi=DPML.generateDB_multi(self.PARAMETERS['n_defects'], self.TEMPERATURE, self.DOPING, self.PARAMETERS['dn_range'], self.PARAM) # two one-level defect data
        db_sah=DPML.generateDB_sah(self.PARAMETERS['n_defects'], self.TEMPERATURE, self.DOPING, self.PARAMETERS['dn_range'], self.PARAM) # one two-level defect data
        db_multi['Mode']=['Two one-level']*len(db_multi)
        db_sah['Mode']=['Single two-level']*len(db_sah)
        dataDf=pd.concat([db_multi,db_sah])
        dataDf['Label']=[0 if mode=="Two one-level" else 1 for mode in dataDf['Mode']]
        dataDf = dataDf.sample(frac=1) # shuffle the generated data.
        self.exp.uploadDB(dataDf)
        vocab={
            '0':'Two one-level',
            '1':'Single two-level',
        }

        # plot the lifetime of the first simulated defect for sanity check
        first_defect = dataDf.iloc[0, :]
        # extract the lifetime data from the first row: whose colunm title starts with a number:
        select_X_list = []
        for string in dataDf.columns.tolist():
            if string[0].isdigit():
                select_X_list.append(string)
        X = first_defect[select_X_list] # take the lifetime as X, delete any column that does not start with a number.

        # now X is hte lifetime data, we need to reshape it into different temperatures: we can read off the line length through the dictionary
        length = len(self.PARAMETERS['dn_range'])

        # reshape X into 100 rows and 6 columes
        lifetimes = np.transpose(np.array(X)).reshape(int(len(X)/length), length)
        # print(np.shape(lifetimes))
        # read off the temperature values from parameters:
        Temp_list = self.TEMPERATURE

        # plot lifetime for different T on a single plot and export it:
        plt.figure(facecolor='white')
        colormap = plt.cm.gist_ncar
        plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(Temp_list)))))
        plt.xscale('log')
        for k in range(len(Temp_list)):
            # print(Temp_list)
            # read off the temperature:
            T = Temp_list[k]
            # read off the lifetime
            lifetime = lifetimes[k, :]
            # plot the graph:
            plt.plot(self.PARAMETERS['dn_range'], lifetime*1e6, label=str(T) + 'K')
        plt.legend()
        plt.xlabel('Excess carrier concentration ' + '$(cm^{-3})$', fontsize=22)
        plt.ylabel('Lifetime ' + '$(\mu s)$', fontsize=22)
        plt.yscale('log')
        # plt.title('Two-level defects lifetime at different Temperature', fontsize=22)
        plt.savefig('lifetime curves of first defect' + '.png')
        plt.show()

        # export the data
        self.exp.exportDataset()

        # # email reminding
        # if emailme:
        #     self.email_reminder()


    def two_level_simulate(self, emailme = False):
        '''
        This function simulate the lifetime for two-level defects.
        '''
        #  Define experiment: every time remember to run this line to refresh the code.
        self.exp = Experiment(SaveDir=self.SAVEDIR, Parameters=self.PARAMETERS)
        # perform the simulation
        # Simualate datasets: for Single two-level, run this section
        # sanity check whether exp is defined.
        db_sah=DPML.generateDB_sah(self.PARAMETERS['n_defects'], self.TEMPERATURE, self.DOPING, self.PARAMETERS['dn_range'], self.PARAM) # one two-level defect data
        # print(db_sah)
        db_sah['Mode']=['Single two-level']*len(db_sah)
        dataDf=db_sah
        dataDf['Label']=[0 if mode=="Two one-level" else 1 for mode in dataDf['Mode']]
        self.exp.uploadDB(dataDf)
        vocab={
            '0':'Two one-level',
            '1':'Single two-level',
        }

        # plot the lifetime of the first simulated defect for sanity check
        first_defect = dataDf.iloc[0, :]
        # extract the lifetime data from the first row: whose colunm title starts with a number:
        select_X_list = []
        for string in dataDf.columns.tolist():
            if string[0].isdigit():
                select_X_list.append(string)
        X = first_defect[select_X_list] # take the lifetime as X, delete any column that does not start with a number.

        # now X is hte lifetime data, we need to reshape it into different temperatures: we can read off the line length through the dictionary
        length = len(self.PARAMETERS['dn_range'])

        # reshape X into 100 rows and 6 columes
        lifetimes = np.transpose(np.array(X)).reshape(int(len(X)/length), length)
        # print(np.shape(lifetimes))
        # read off the temperature values from parameters:
        Temp_list = self.TEMPERATURE

        # plot lifetime for different T on a single plot and export it:
        plt.figure(facecolor='white')
        colormap = plt.cm.gist_ncar
        plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(Temp_list)))))
        plt.xscale('log')
        for k in range(len(Temp_list)):
            # print(Temp_list)
            # read off the temperature:
            T = Temp_list[k]
            # read off the lifetime
            lifetime = lifetimes[k, :]
            # plot the graph:
            plt.plot(self.PARAMETERS['dn_range'], lifetime*1e6, label=str(T) + 'K')
        plt.legend()
        plt.xlabel('Excess carrier concentration ' + '$(cm^{-3})$', fontsize=22)
        plt.ylabel('Lifetime ' + '$(\mu s)$', fontsize=22)
        plt.yscale('log')
        # plt.title('Two-level defects lifetime at different Temperature', fontsize=22)
        plt.savefig('lifetime curves of first defect' + '.png')
        plt.show()

        # export the data
        self.exp.exportDataset()

        if emailme:
            self.email_reminder()
    

    def two_level_simulate_BO(self, emailme = False):
        '''
        This function simulate the lifetime for two-level defects specifically for BO
        '''
        #  Define experiment: every time remember to run this line to refresh the code.
        self.exp = Experiment(SaveDir=self.SAVEDIR, Parameters=self.PARAMETERS)

        # update the parameters to match with BO measurement
        self.PARAM={
        # 'type': self.WAFERTYPE,                #   Wafer doping type
        # 'Et_min_1':-0.55,             #   Minimum defect energy level
        # 'Et_max_1':0.55,              #   Maximum defect energy level
        # 'Et_min_2':-0.55,             #   Minimum defect energy level
        # 'Et_max_2':0.55,              #   Maximum defect energy level
        # 'S_min_1_p':10**(-17),              #   Minimum capture cross section for hole.
        # 'S_min_1_n':10**(-17),          #   Minimum capture cross section for electron.
        # 'S_max_1_p':10**(-13),              #   Maximum capture cross section for hole.
        # 'S_max_1_n':10**(-13),              # maximum capcture cross section for electron.
        # 'S_min_2_p':1E-17,              #   Minimum capture cross section for hole.
        # 'S_min_2_n':1E-17,          #   Minimum capture cross section for electron.
        # 'S_max_2_p':1E-13,              #   Maximum capture cross section for hole.
        # 'S_max_2_n':1E-13,              # maximum capcture cross section for electron.
        # the parameter to simulate BO:
        'Et_min_1':0.15,             #   Minimum defect energy level
        'Et_max_1':0.15,              #   Maximum defect energy level
        'Et_min_2':-0.3,             #   Minimum defect energy level
        'Et_max_2':-0.3,              #   Maximum defect energy level
        'S_min_1_p':1e-14/18,              #   Minimum capture cross section for hole.
        'S_min_1_n':1e-14,          #   Minimum capture cross section for electron.
        'S_max_1_p':1e-14/18,              #   Maximum capture cross section for hole.
        'S_max_1_n':1e-14,              # maximum capcture cross section for electron.
        'S_min_2_p':1e-14/103.2*86,              #   Minimum capture cross section for hole.
        'S_min_2_n':1e-14/103.2,          #   Minimum capture cross section for electron.
        'S_max_2_p':1e-14/103.2*86,              #   Maximum capture cross section for hole.
        'S_max_2_n':1E-14/103.2,              # maximum capcture cross section for electron.
        'Nt':5E10,                  #   Defect density
        'check_auger': False,     #   Check wether to resample if lifetime is auger-limited
        'noise':'',             #   Enable noiseparam
        'noiseparam':0,         #   Adds noise proportional to the log of Delta n
        }

        # Simualate datasets: for p type 303K and excess carrier concentration from 1e13 to 3e15
        # update the wafer type
        self.WAFERTYPE = 'p'
        self.PARAM['type'] = self.WAFERTYPE
        # update the defect density
        self.PARAM['Nt'] = 3e10
        # update the excess carrier concentration
        self.PARAMETERS['dn_range'] = np.logspace(13,15.5,100)
        # simulate the data
        db_sah = DPML.generateDB_sah(self.PARAMETERS['n_defects'], [303], [8.5e14], np.logspace(13,15.5,100), self.PARAM) # one two-level defect data
        db_sah['Mode']=['Single two-level']*len(db_sah)
        dataDf_p_303 = db_sah
        dataDf_p_303['Label']=[0 if mode=="Two one-level" else 1 for mode in dataDf_p_303['Mode']]


        # Simualate datasets: for p type 343K and excess carrier concentration from 1e14 to 3e16
        # update the wafer type
        # self.WAFERTYPE = 'n'
        # self.PARAM['type'] = self.WAFERTYPE
        # # update the defect density
        # self.PARAM['Nt'] = 2E12
        # # update the excess carrier concentration
        # self.PARAMETERS['dn_range'] = np.logspace(14,16.5,100)
        # # simulate the data
        # db_sah=DPML.generateDB_sah(self.PARAMETERS['n_defects'], [343], [2e15], np.logspace(14,16.5,100), self.PARAM) # one two-level defect data
        # db_sah['Mode']=['Single two-level']*len(db_sah)
        # dataDf_n_343=db_sah
        # dataDf_n_343['Label']=[0 if mode=="Two one-level" else 1 for mode in dataDf_n_343['Mode']]


         # Simualate datasets: for n type 303K and excess carrier concentration from 1e14 to 3e16
        # update the wafer type
        self.WAFERTYPE = 'n'
        self.PARAM['type'] = self.WAFERTYPE
        # update the defect density
        self.PARAM['Nt'] = 1E12
        # update the excess carrier concentration
        self.PARAMETERS['dn_range'] = np.logspace(14,16.5,100)
        # simulate the data
        db_sah=DPML.generateDB_sah(self.PARAMETERS['n_defects'], [303], [2e15], np.logspace(14,16.5,100), self.PARAM) # one two-level defect data
        db_sah['Mode']=['Single two-level']*len(db_sah)
        dataDf_n_303=db_sah
        dataDf_n_303['Label']=[0 if mode=="Two one-level" else 1 for mode in dataDf_n_303['Mode']]


        # combine the three lifetime curves horizontally
        # dataDf = pd.concat([dataDf_p_303, dataDf_n_343, dataDf_n_303],axis=1)
        dataDf = pd.concat([dataDf_p_303, dataDf_n_303],axis=1)
        # remove the duplicated columns
        dataDf = dataDf.loc[:,~dataDf.columns.duplicated()]
        # upload the data
        self.exp.uploadDB(dataDf)
        vocab={
        '0':'Two one-level',
        '1':'Single two-level',
        }

        # export the data
        self.exp.exportDataset()
        
        # # plot the lifetime of the first simulated defect for sanity check
        # first_defect = dataDf.iloc[0, :]
        # # extract the lifetime data from the first row: whose colunm title starts with a number:
        # select_X_list = []
        # for string in dataDf.columns.tolist():
        #     if string[0].isdigit():
        #         select_X_list.append(string)
        # X = first_defect[select_X_list] # take the lifetime as X, delete any column that does not start with a number.

        # # now X is hte lifetime data, we need to reshape it into different temperatures: we can read off the line length through the dictionary
        # length = len(self.PARAMETERS['dn_range'])

        # # reshape X into 100 rows and 6 columes
        # lifetimes = np.transpose(np.array(X)).reshape(int(len(X)/length), length)
        # # read off the temperature values from parameters:
        # Temp_list = [303, 343, 303]

#         # plot lifetime for different T on a single plot and export it:
#         plt.figure(facecolor='white')
#         colormap = plt.cm.gist_ncar
#         plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(Temp_list)))))
#         plt.xscale('log')
#         # plot the p type 303K: 1e6 is to convert from s to us
#         plt.plot(np.logspace(13, 15.5, 100), lifetimes[0, :]*1e6, label='303K, p-type')
#         # plot with Yan data
# #         plt.plot([27534022784368,
# # 31550792957346.9,
# # 36153545162406.1,
# # 41427764734064.4,
# # 47471407939424.3,
# # 54396721286250.3,
# # 62332326238771,
# # 71425608059923.3,
# # 81845453147175,
# # 93785385701533.1,
# # 107467161986003,
# # 123144888928437,
# # 141109743562151,
# # 161695380956878,
# # 185284131079681,
# # 212314099678011,
# # 243287305067151,
# # 278779001943822,
# # 319448365393920,
# # 366050733524800,
# # 419451636100292,
# # 480642869727488,
# # 550760918154195,
# # 631108059790913,
# # 723176554480104,
# # 828676358725287,
# # 949566884125385,
# # 1088093388852790,
# # 1246828678061630,
# # 1428720887713460,
# # 1637148239293100,
# # 1875981782354990,
# # 2149657290195900,
# # 2463257643947610,
# # 2822607235180910
# # ], np.array([0.000601262740429958,
# # 0.000658665531867146,
# # 0.000718181702553165,
# # 0.000776728809090367,
# # 0.000827604665586884,
# # 0.000889510134291615,
# # 0.000950120788351192,
# # 0.000983243203324339,
# # 0.000980289928468689,
# # 0.00109167977048749,
# # 0.00123483478114777,
# # 0.00133801501230349,
# # 0.00142922981842247,
# # 0.00153012157703939,
# # 0.00171956018953192,
# # 0.00173043052728484,
# # 0.00188110130147385,
# # 0.0019827315220447,
# # 0.00204251291116385,
# # 0.002171575705917,
# # 0.00234573842998269,
# # 0.00243317703947774,
# # 0.00245459042800956,
# # 0.00251241177161198,
# # 0.00266486217885697,
# # 0.00266192674623504,
# # 0.00272059528039267,
# # 0.00288529927412184,
# # 0.00293309196337702,
# # 0.00289984139428033,
# # 0.0029691330081956,
# # 0.00300099571696042,
# # 0.00296425092032784,
# # 0.00300583674711878,
# # 0.00299323759568535,
# # ])*1e6, label='303K, p-type, Yan data')
#         # plot the p type 343K: 1e6 is to convert from s to us
#         plt.plot(np.logspace(14, 16.5, 100), lifetimes[1, :]*1e6, label='343K, n-type')
#         plt.plot([103187160115410,
# 112716917530961,
# 123126787126145,
# 134498050870162,
# 146919497455412,
# 160488115574167,
# 175309851222190,
# 191500434942445,
# 209186285468160,
# 228505496821028,
# 249608916571983,
# 272661323683771,
# 297842715132119,
# 325349711351646,
# 355397091480464,
# 388219470390916,
# 424073130600979,
# 463238023370204,
# 506019954605061,
# 552752972641620,
# 603801976549742,
# 659565565324876,
# 720479150214384,
# 787018354479978,
# 859702727142161,
# 939099799704190,
# 1025829517531140,
# 1120569080485000,
# 1224058230612260,
# 1337105028171170,
# 1460592161098680,
# 1595483837182800,
# 1742833312754940,
# 1903791116688320,
# 2079614033915900,
# 2271674918613470,
# 2481473413670080,
# 2710647660145780,
# 2960987088145640,
# 3234446388983540,
# 3533160777732550,
# 3859462665334400,
# 4215899870446730,
# 4605255513229000,
# 5030569746401210,
# 5495163493254960,
# 6002664377967210,
# 6557035050684830,
# 7162604128546910,
# 7824099994237340,
# 8546687715972350,
# 9336009377200320,
# 10198228130914700,
# 11140076322563500,
# 12168908057306600,
# 13292756622075600,
# 14520397210794800,
# 15861415442535600,
# 17326282207602000,
# 18926435425959500,
# 20674369356390000,
# 22583732153714200,
# 24669432435822800,
# ],np.array([0.000348695673907163,
# 0.000279921259379822,
# 0.000313862430652565,
# 0.000255042067306108,
# 0.000281305940339077,
# 0.00033007376479769,
# 0.000297656585383869,
# 0.000300313968836968,
# 0.000262957265597958,
# 0.000274404603619128,
# 0.000299781708818235,
# 0.000288108710028952,
# 0.000264378878545744,
# 0.000268657042758276,
# 0.000266069901834315,
# 0.000250112044120745,
# 0.000262857274637482,
# 0.000258346763800657,
# 0.000247299168078695,
# 0.000244965508926453,
# 0.000238261912360439,
# 0.000237764258977961,
# 0.000232446557912611,
# 0.000224376428250502,
# 0.000216949451532133,
# 0.000219058490523047,
# 0.000215827138330216,
# 0.000211994738667066,
# 0.000207994854377211,
# 0.000208634885881568,
# 0.000202198629604142,
# 0.000200437506753696,
# 0.00019718472744064,
# 0.000196233146769916,
# 0.000191958298465989,
# 0.000189884088006154,
# 0.000190179253627594,
# 0.000187644219733863,
# 0.00018621695005858,
# 0.000184399107272355,
# 0.000183754410809022,
# 0.000181935706926214,
# 0.000180363678878678,
# 0.000179180695273686,
# 0.000178375080510873,
# 0.000177347696132585,
# 0.000176940086271837,
# 0.000176442278268817,
# 0.000176488737810985,
# 0.000176616414287653,
# 0.000176724299175871,
# 0.000176387149168174,
# 0.000176147372953221,
# 0.000175968818192952,
# 0.000176617038046456,
# 0.000176817267808765,
# 0.000177253732061947,
# 0.000177852249515638,
# 0.000178411485205063,
# 0.000179228578622751,
# 0.000180091525205379,
# 0.000181200559918785,
# 0.000180869344610297,
# ])*1e6, label='343K, n-type, Yan data')
#         # plot the n type 303K: 1e6 is to convert from s to us
#         plt.plot(np.logspace(14, 16.5, 100), lifetimes[2, :]*1e6, label='303K, n-type')
#         plt.plot([117542045198389,
# 128978238867442,
# 141527111198971,
# 155296919699079,
# 170406454733019,
# 186986064314375,
# 205178778600603,
# 225141543795779,
# 247046576105301,
# 271082847422922,
# 297457715566948,
# 326398713129446,
# 358155510370618,
# 393002069091997,
# 431239006069643,
# 473196186436365,
# 519235569385757,
# 569754330747564,
# 625188289372441,
# 686015666885052,
# 752761213240487,
# 826000733674551,
# 906366056101416,
# 994550481811691,
# 1091314766493410,
# 1197493683173400,
# 1314003223696740,
# 1441848500870470,
# 1582132419442390,
# 1736065190718320,
# 1904974772899290,
# 2090318327205880,
# 2293694788620600,
# 2516858659694030,
# 2761735146412570,
# 3030436766702530,
# 3325281574850400,
# 3648813159058730,
# 4003822584654080,
# 4393372472248330,
# 4820823418572380,
# 5289862987911570,
# 5804537524247930,
# 6369287058548970,
# 6988983612343110,
# 7668973228022220,
# 8415122088461170,
# 9233867123822200,
# 10132271542122900,
# 11118085762622200,
# 12199814277690000,
# 13386789019967500,
# 14689249867749300,
# 16118432983094500,
# 17686667744751000,
# 19407483112118200,
# 21295724337838300,
# 23367681035879500,
# 25641227709939200,
# ],np.array([0.000149384864464721,
# 0.000158357872845068,
# 0.000149607268649753,
# 0.000150241353019561,
# 0.000158521950205039,
# 0.000149555492483859,
# 0.000157466747952028,
# 0.000155645975644549,
# 0.000153205176128088,
# 0.000154448216428553,
# 0.000152528465847741,
# 0.000147887629728663,
# 0.000144695529021539,
# 0.000145376340809639,
# 0.000149742703021257,
# 0.000147641350414667,
# 0.000141002343244598,
# 0.000140083943671049,
# 0.000140936981403673,
# 0.000139967833982669,
# 0.000138603822287922,
# 0.000136697895298875,
# 0.000137674547424659,
# 0.000135595150661463,
# 0.00013482495406114,
# 0.000134916088640091,
# 0.000134127594220305,
# 0.00013450507766738,
# 0.000134763608200232,
# 0.000134529204678436,
# 0.000133958963800213,
# 0.000134693006295653,
# 0.000134552692201467,
# 0.000134647330875716,
# 0.000134933255807573,
# 0.000135549989280338,
# 0.00013624189030023,
# 0.00013623431408167,
# 0.000136358353839015,
# 0.000136394967887297,
# 0.000137606800600332,
# 0.000137845260831909,
# 0.000138498047210152,
# 0.000139283596650858,
# 0.000140305444927538,
# 0.00014142582201297,
# 0.000142192773028039,
# 0.000143099669323422,
# 0.000144138104499054,
# 0.000145153177374482,
# 0.000146098255761071,
# 0.000147599643835638,
# 0.000148613356920093,
# 0.000149818436393802,
# 0.000151109143237704,
# 0.000152411563988999,
# 0.000153877233412181,
# 0.000155072782534966,
# 0.00015167544798772,
# ])*1e6, label='303K, n-type, Yan data')

#         plt.legend()
#         plt.xlabel('Excess carrier concentration ' + '$(cm^{-3})$', fontsize=22)
#         plt.ylabel('Lifetime ' + '$(\mu s)$', fontsize=22)
#         plt.yscale('log')
#         # plt.title('Two-level defects lifetime at different Temperature', fontsize=22)
#         plt.savefig('lifetime curves of first defect' + '.png')
#         plt.show()

        if emailme:
            self.email_reminder()


    def BO_training_pn_303K(self, emailme = False):
        '''
        This function simulate the lifetime for two-level defects specifically for BO
        '''
        #  Define experiment: every time remember to run this line to refresh the code.
        self.exp = Experiment(SaveDir=self.SAVEDIR, Parameters=self.PARAMETERS)

        # update the parameters to match with BO measurement
        self.PARAM={
        # 'type': self.WAFERTYPE,                #   Wafer doping type
        # 'Et_min_1':-0.55,             #   Minimum defect energy level
        # 'Et_max_1':0.55,              #   Maximum defect energy level
        # 'Et_min_2':-0.55,             #   Minimum defect energy level
        # 'Et_max_2':0.55,              #   Maximum defect energy level
        # 'S_min_1_p':10**(-17),              #   Minimum capture cross section for hole.
        # 'S_min_1_n':10**(-17),          #   Minimum capture cross section for electron.
        # 'S_max_1_p':10**(-13),              #   Maximum capture cross section for hole.
        # 'S_max_1_n':10**(-13),              # maximum capcture cross section for electron.
        # 'S_min_2_p':1E-17,              #   Minimum capture cross section for hole.
        # 'S_min_2_n':1E-17,          #   Minimum capture cross section for electron.
        # 'S_max_2_p':1E-13,              #   Maximum capture cross section for hole.
        # 'S_max_2_n':1E-13,              # maximum capcture cross section for electron.
        # the parameter to simulate BO:
        'Et_min_1':0,             #   Minimum defect energy level
        'Et_max_1':0.55,              #   Maximum defect energy level
        'Et_min_2':-0.55,             #   Minimum defect energy level
        'Et_max_2':0,              #   Maximum defect energy level
        'S_min_1_p':1e-14/18,              #   Minimum capture cross section for hole.
        'S_min_1_n':1e-14,          #   Minimum capture cross section for electron.
        'S_max_1_p':1e-14/18,              #   Maximum capture cross section for hole.
        'S_max_1_n':1e-14,              # maximum capcture cross section for electron.
        'S_min_2_p':1e-14/103.2*86,              #   Minimum capture cross section for hole.
        'S_min_2_n':1e-14/103.2,          #   Minimum capture cross section for electron.
        'S_max_2_p':1e-14/103.2*86,              #   Maximum capture cross section for hole.
        'S_max_2_n':1E-14/103.2,              # maximum capcture cross section for electron.
        'Nt':5E10,                  #   Defect density
        'check_auger': False,     #   Check wether to resample if lifetime is auger-limited
        'noise':'',             #   Enable noiseparam
        'noiseparam':0,         #   Adds noise proportional to the log of Delta n
        }

        # Simualate datasets: for p type 303K and excess carrier concentration from 1e13 to 3e15
        # update the wafer type
        self.WAFERTYPE = 'p'
        self.PARAM['type'] = self.WAFERTYPE
        # update the defect density
        self.PARAM['Nt'] = 3e10
        # update the excess carrier concentration
        self.PARAMETERS['dn_range'] = np.logspace(13,15.5,100)
        # simulate the data
        db_sah = DPML.generateDB_sah(self.PARAMETERS['n_defects'], [303], [8.5e14], np.logspace(13,15.5,100), self.PARAM) # one two-level defect data
        db_sah['Mode']=['Single two-level']*len(db_sah)
        dataDf_p_303 = db_sah
        dataDf_p_303['Label']=[0 if mode=="Two one-level" else 1 for mode in dataDf_p_303['Mode']]

         # Simualate datasets: for n type 303K and excess carrier concentration from 1e14 to 3e16
        # update the wafer type
        self.WAFERTYPE = 'n'
        self.PARAM['type'] = self.WAFERTYPE
        # update the defect density
        self.PARAM['Nt'] = 1E12
        # update the excess carrier concentration
        self.PARAMETERS['dn_range'] = np.logspace(14,16.5,100)
        # simulate the data
        db_sah=DPML.generateDB_sah(self.PARAMETERS['n_defects'], [303], [2e15], np.logspace(14,16.5,100), self.PARAM) # one two-level defect data
        db_sah['Mode']=['Single two-level']*len(db_sah)
        dataDf_n_303=db_sah
        dataDf_n_303['Label']=[0 if mode=="Two one-level" else 1 for mode in dataDf_n_303['Mode']]


        # combine the three lifetime curves horizontally
        dataDf = pd.concat([dataDf_p_303, dataDf_n_303],axis=1)
        # remove the duplicated columns
        dataDf = dataDf.loc[:,~dataDf.columns.duplicated()]
        # upload the data
        self.exp.uploadDB(dataDf)
        vocab={
        '0':'Two one-level',
        '1':'Single two-level',
        }


        # plot the lifetime of the first simulated defect for sanity check
        first_defect = dataDf.iloc[0, :]
        # extract the lifetime data from the first row: whose colunm title starts with a number:
        select_X_list = []
        for string in dataDf.columns.tolist():
            if string[0].isdigit():
                select_X_list.append(string)
        X = first_defect[select_X_list] # take the lifetime as X, delete any column that does not start with a number.

        # now X is hte lifetime data, we need to reshape it into different temperatures: we can read off the line length through the dictionary
        length = len(self.PARAMETERS['dn_range'])

        # reshape X into 100 rows and 6 columes
        lifetimes = np.transpose(np.array(X)).reshape(int(len(X)/length), length)
        # read off the temperature values from parameters:
        Temp_list = [303, 343, 303]

        # plot lifetime for different T on a single plot and export it:
        plt.figure(facecolor='white')
        colormap = plt.cm.gist_ncar
        plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(Temp_list)))))
        plt.xscale('log')
        # plot the p type 303K: 1e6 is to convert from s to us
        plt.plot(np.logspace(13, 15.5, 100), lifetimes[0, :]*1e6, label='303K, p-type')
        # plot the p type 343K: 1e6 is to convert from s to us
        plt.plot(np.logspace(14, 16.5, 100), lifetimes[1, :]*1e6, label='343K, n-type')
        # plot the n type 303K: 1e6 is to convert from s to us
        # plt.plot(np.logspace(14, 16.5, 100), lifetimes[2, :]*1e6, label='303K, n-type')

        plt.legend()
        plt.xlabel('Excess carrier concentration ' + '$(cm^{-3})$', fontsize=22)
        plt.ylabel('Lifetime ' + '$(\mu s)$', fontsize=22)
        plt.yscale('log')
        # plt.title('Two-level defects lifetime at different Temperature', fontsize=22)
        plt.savefig('lifetime curves of first defect' + '.png')
        plt.show()

        # export the data
        self.exp.exportDataset()

        if emailme:
            self.email_reminder()


    def experimental_data_interpolator(self):
        '''
        Interpolates the experimental data measured by Yan into the same excess 
        carrier concentration as the simulated data
        '''

        # Interpolate the p type 303K, 8.5e14 cm-3 dn into logspace(13, 15.5, 100)
        new_dn_values_p_303_8_5e14 = np.logspace(13, 15.5, 100)
        interpolated_tau_values_p_303_8_5e14 = np.interp(new_dn_values_p_303_8_5e14, 
                                                        self.dn_p_303_8_5e14, 
                                                        self.tau_p_303_8_5e14)
        # get rid of the ripple thorugh filttering: we know Sah Shockley does not give ripple
        interpolated_tau_values_p_303_8_5e14 = savgol_filter(interpolated_tau_values_p_303_8_5e14, 31, 2)
        # store into the object
        self.dn_p_303_8_5e14_interpolated = new_dn_values_p_303_8_5e14
        self.tau_p_303_8_5e14_interpolated = interpolated_tau_values_p_303_8_5e14
        # Interpolate the p type 343K, 2e15 cm-3 dn into logspace(14, 16.5, 100)
        new_dn_values_p_343_2e15 = np.logspace(14, 16.5, 100)
        interpolated_tau_values_p_343_2e15 = np.interp(new_dn_values_p_343_2e15, 
                                                    self.dn_p_343_2e15, 
                                                    self.tau_p_343_2e15)
        interpolated_tau_values_p_343_2e15 = savgol_filter(interpolated_tau_values_p_343_2e15, 71, 2)
        self.dn_p_343_2e15_interpolated = new_dn_values_p_343_2e15
        self.tau_p_343_2e15_interpolated = interpolated_tau_values_p_343_2e15
        # Interpolate the n type 303K, 2e15 cm-3 dn into logspace(14, 16.5, 100)
        new_dn_values_n_303_2e15 = np.logspace(14, 16.5, 100)
        interpolated_tau_values_n_303_2e15 = np.interp(new_dn_values_n_303_2e15, 
                                                    self.dn_n_303_2e15, 
                                                    self.tau_n_303_2e15)
        interpolated_tau_values_n_303_2e15 = savgol_filter(interpolated_tau_values_n_303_2e15, 19, 2)   
        self.dn_n_303_2e15_interpolated = new_dn_values_n_303_2e15
        self.tau_n_303_2e15_interpolated = interpolated_tau_values_n_303_2e15

        # Plot
        plt.figure(facecolor='white')
        # p type 303K, 8.5e14
        plt.plot(new_dn_values_p_303_8_5e14, interpolated_tau_values_p_303_8_5e14, label='p type 303K, 8.5e14')
        plt.scatter(self.dn_p_303_8_5e14, self.tau_p_303_8_5e14, s=10)
        
        # p type 343K, 2e15
        plt.plot(new_dn_values_p_343_2e15, interpolated_tau_values_p_343_2e15, label='p type 343K, 2e15')
        plt.scatter(self.dn_p_343_2e15, self.tau_p_343_2e15, s=10)
        
        # n type 303K, 2e15
        plt.plot(new_dn_values_n_303_2e15, interpolated_tau_values_n_303_2e15, label='n type 303K, 2e15')
        plt.scatter(self.dn_n_303_2e15, self.tau_n_303_2e15, s=10)
        
        plt.legend()
        plt.xlabel('Excess carrier concentration ' + '$(cm^{-3})$', fontsize=22)
        plt.ylabel('Lifetime ' + '$(s)$', fontsize=22)
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Raw Lifetime Data', fontsize=22)
        plt.show()

        # now export the data the same format as the simulator


    def interpolate_data_exporter(self):
        '''
        1. Simulate the BO lifeitme data.
        2. Replace the BO lifetime data with interpolated experimental data.
        3. Export the interpolated experimentla data in the format of simuolated data.
        '''
        # Run the BO simulation
        #  Define experiment: every time remember to run this line to refresh the code.
        self.exp = Experiment(SaveDir=self.SAVEDIR, Parameters=self.PARAMETERS)

        # update the parameters to match with BO measurement
        self.PARAM={
        # 'type': self.WAFERTYPE,                #   Wafer doping type
        # 'Et_min_1':-0.55,             #   Minimum defect energy level
        # 'Et_max_1':0.55,              #   Maximum defect energy level
        # 'Et_min_2':-0.55,             #   Minimum defect energy level
        # 'Et_max_2':0.55,              #   Maximum defect energy level
        # 'S_min_1_p':10**(-17),              #   Minimum capture cross section for hole.
        # 'S_min_1_n':10**(-17),          #   Minimum capture cross section for electron.
        # 'S_max_1_p':10**(-13),              #   Maximum capture cross section for hole.
        # 'S_max_1_n':10**(-13),              # maximum capcture cross section for electron.
        # 'S_min_2_p':1E-17,              #   Minimum capture cross section for hole.
        # 'S_min_2_n':1E-17,          #   Minimum capture cross section for electron.
        # 'S_max_2_p':1E-13,              #   Maximum capture cross section for hole.
        # 'S_max_2_n':1E-13,              # maximum capcture cross section for electron.
        # the parameter to simulate BO:
        'Et_min_1':0.15,             #   Minimum defect energy level
        'Et_max_1':0.15,              #   Maximum defect energy level
        'Et_min_2':-0.3,             #   Minimum defect energy level
        'Et_max_2':-0.3,              #   Maximum defect energy level
        'S_min_1_p':1e-14/18,              #   Minimum capture cross section for hole.
        'S_min_1_n':1e-14,          #   Minimum capture cross section for electron.
        'S_max_1_p':1e-14/18,              #   Maximum capture cross section for hole.
        'S_max_1_n':1e-14,              # maximum capcture cross section for electron.
        'S_min_2_p':1e-14/103.2*86,              #   Minimum capture cross section for hole.
        'S_min_2_n':1e-14/103.2,          #   Minimum capture cross section for electron.
        'S_max_2_p':1e-14/103.2*86,              #   Maximum capture cross section for hole.
        'S_max_2_n':1E-14/103.2,              # maximum capcture cross section for electron.
        'Nt':5E10,                  #   Defect density
        'check_auger': False,     #   Check wether to resample if lifetime is auger-limited
        'noise':'',             #   Enable noiseparam
        'noiseparam':0,         #   Adds noise proportional to the log of Delta n
        }

        # Simualate datasets: for p type 303K and excess carrier concentration from 1e13 to 3e15
        # update the wafer type
        self.WAFERTYPE = 'p'
        self.PARAM['type'] = self.WAFERTYPE
        # update the defect density
        self.PARAM['Nt'] = 3e10
        # update the excess carrier concentration
        self.PARAMETERS['dn_range'] = np.logspace(13,15.5,100)
        # simulate the data
        db_sah = DPML.generateDB_sah(self.PARAMETERS['n_defects'], [303], [8.5e14], np.logspace(13,15.5,100), self.PARAM) # one two-level defect data
        db_sah['Mode']=['Single two-level']*len(db_sah)
        dataDf_p_303 = db_sah
        dataDf_p_303['Label']=[0 if mode=="Two one-level" else 1 for mode in dataDf_p_303['Mode']]


        # Simualate datasets: for p type 343K and excess carrier concentration from 1e14 to 3e16
        # update the wafer type
        # self.WAFERTYPE = 'n'
        # self.PARAM['type'] = self.WAFERTYPE
        # # update the defect density
        # self.PARAM['Nt'] = 2E12
        # # update the excess carrier concentration
        # self.PARAMETERS['dn_range'] = np.logspace(14,16.5,100)
        # # simulate the data
        # db_sah=DPML.generateDB_sah(self.PARAMETERS['n_defects'], [343], [2e15], np.logspace(14,16.5,100), self.PARAM) # one two-level defect data
        # db_sah['Mode']=['Single two-level']*len(db_sah)
        # dataDf_n_343=db_sah
        # dataDf_n_343['Label']=[0 if mode=="Two one-level" else 1 for mode in dataDf_n_343['Mode']]


         # Simualate datasets: for n type 303K and excess carrier concentration from 1e14 to 3e16
        # update the wafer type
        self.WAFERTYPE = 'n'
        self.PARAM['type'] = self.WAFERTYPE
        # update the defect density
        self.PARAM['Nt'] = 1E12
        # update the excess carrier concentration
        self.PARAMETERS['dn_range'] = np.logspace(14,16.5,100)
        # simulate the data
        db_sah=DPML.generateDB_sah(self.PARAMETERS['n_defects'], [303], [2e15], np.logspace(14,16.5,100), self.PARAM) # one two-level defect data
        db_sah['Mode']=['Single two-level']*len(db_sah)
        dataDf_n_303=db_sah
        dataDf_n_303['Label']=[0 if mode=="Two one-level" else 1 for mode in dataDf_n_303['Mode']]


        # combine the three lifetime curves horizontally
        # dataDf = pd.concat([dataDf_p_303, dataDf_n_343, dataDf_n_303],axis=1)
        dataDf = pd.concat([dataDf_p_303, dataDf_n_303],axis=1)
        # remove the duplicated columns
        dataDf = dataDf.loc[:,~dataDf.columns.duplicated()]

        # replace the simulation with interpolated experimental data.
        # extract the lifetime data: whose colunm title starts with a number:
        select_X_list = []
        for string in dataDf.columns.tolist():
            if string[0].isdigit():
                select_X_list.append(string)
        # replace the simulation with interpolated experimental data.
        interpolatedata = np.concatenate((self.tau_p_303_8_5e14_interpolated.reshape(1,100),  
                                                self.tau_n_303_2e15_interpolated.reshape(1,100)), axis=1)
        dataDf[select_X_list] = interpolatedata

        # now X is hte lifetime data, we need to reshape it into different temperatures: we can read off the line length through the dictionary

        # upload the data
        self.exp.uploadDB(dataDf)
        vocab={
        '0':'Two one-level',
        '1':'Single two-level',
        }

        # export the data
        self.exp.exportDataset()


    def email_reminder(self):
        '''
        This function sends email to myself to remind me when the simulation is done
        '''
        # who to send to. and the content of the email.
        # email title
        subject='data generation done'
        # email body
        body= 'data generation is done' + ' through the file ' + str(os.getcwd())
        # which email address to sent to:
        to='z5183876@ad.unsw.edu.au'

        # email address to be sent from: (you can use this address to send email from)
        user = "sijinwang@yahoo.com"
        password = 'gdfkzhzhaokjivek'

        msg = EmailMessage()
        msg.set_content(body)
        msg['subject'] = subject
        msg['to'] = to
        msg['from'] = user

        server = smtplib.SMTP("smtp.mail.yahoo.com", 587)
        server.starttls()
        server.login(user, password)
        server.send_message(msg)

        server.quit()