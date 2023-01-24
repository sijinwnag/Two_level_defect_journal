# %%-- Imports
import sys
# sys.path.append(r'C:\Users\budac\Documents\GitHub\Yoann_code\DPML')
# sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\Yoann_code\DPML')
# from Si import *

# use this line for dell laptop:
# sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\yoann_code_new')
# use this line for workstation:
sys.path.append(r'C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\Two_level_defect_journal\simulation\DPML')
sys.path.append(r'C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\Two_level_defect_journal\simulation')
from DPML import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import smtplib
from email.message import EmailMessage
import os

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
        self.SAVEDIR = r"C:\Users\sijin wang\Documents\GitHub\Two_level_defect_journal\simulation\simulated_data"
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
        #  Define experiment: every time remember to run this line to refresh the code.
        self.exp = Experiment(SaveDir=self.SAVEDIR, Parameters=self.PARAMETERS)


    def mixture_simulate(self, emailme = False):
        '''
        This function simulate the lifetime for a mixture of one-level and two-level defects.
        '''

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