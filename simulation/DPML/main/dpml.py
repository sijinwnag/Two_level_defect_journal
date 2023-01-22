"Main functions for DPML"
import sys
# import the function file from another folder:
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\yoann_code_new\DPML')
from Si import Cell,Defect,LTS, Sah
from main import ML
from utils.matplotlibstyle import *
from utils import SaveObj, LoadObj
from utils import Logger
import numpy as np
import os
import warnings
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import warnings
from decimal import Decimal
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\    TO DO
#   Add logbook as .log file
#   Add object saving as .dpml file one object per saveDir
#   Local default parameter per function
#   Revamp noise model to match format from T-CCS




class DPML():
    #****   constant declaration    ****#
    DefaultParameters = {
        'save': False,
        'generateDB':{
                'type': 'p',                #   Wafer doping type
                'Et_min':-0.55,             #   Minimum defect energy level
                'Et_max':0.55,              #   Maximum defect energy level
                'S_min_n':1E-18,              #   Minimum capture cross section for electron.
                'S_min_p':1E-18,             # minimum capture cross section for hole
                'S_max_n':1E-12,              #   Maximum capture cross section for electron.
                'S_max_p':1E-12,            # maximum capture cross section for hole.
                'Nt':1E12,                  #   Defect density
                'CMn_tab':['Radiative'],    #   Capture mode for Sn
                'CMp_tab':['Radiative'],    #   Capture mode for Sp
                'Force_same_CM':False,      #   Wether to force Sn and Sp to follow CMn_tab
                'check_auger':True,        #   Check wether to resample if lifetime is auger-limited
                'noise':'',     #   Enable noiseparam
                'noiseparam':0,     #   Adds noise proportional to the log of Delta n
        },
        'generateDB_multi':{
                'type': 'p',                #   Wafer doping type
                'Et_min_1':-0.55,             #   Minimum defect energy level
                'Et_max_1':0.55,              #   Maximum defect energy level
                'Et_min_2':-0.55,             #   Minimum defect energy level
                'Et_max_2':0.55,              #   Maximum defect energy level
                'S_min_1_p':1E-18,              #   Minimum capture cross section for hole for defect 1
                'S_min_1_n':1E-18,          # Minimum capture cross section for electron for defect 1
                'S_max_1_p':1E-12,              #   Maximum capture cross section for hole for defect 1
                'S_max_1_n':1E-12,          # Maximum capture cross section for electron for defect 1
                'S_min_2_p':1E-18,              #   Minimum capture cross section for hole for defect 2
                'S_min_2_n':1E-18,          # Minimum capcture cross section for electron for defect 2
                'S_max_2_p':1E-12,              #   Maximum capture cross section for hole for defect 2
                'S_max_2_n':1E-12,          # Maximum capture cross section for electron for defect 2
                'Nt_1':1E12,                  #   Defect density
                'Nt_2':1E12,                  #   Defect density
                'check_auger':True,        #   Check wether to resample if lifetime is auger-limited
                'noise':'',     #   Enable noiseparam
                'noiseparam':0,     #   Adds noise proportional to the log of Delta n
        },
        'generateDB_sah':{
                'type': 'p',                #   Wafer doping type
                'Et_min_1':-0.55,             #   Minimum defect energy level
                'Et_max_1':0.55,              #   Maximum defect energy level
                'Et_min_2':-0.55,             #   Minimum defect energy level
                'Et_max_2':0.55,              #   Maximum defect energy level
                'S_min_1_p':1E-18,              #   Minimum capture cross section for hole for defect 1
                'S_min_1_n':1E-18,          # Minimum capture cross section for electron for defect 1
                'S_max_1_p':1E-12,              #   Maximum capture cross section for hole for defect 1
                'S_max_1_n':1E-12,          # Maximum capture cross section for electron for defect 1
                'S_min_2_p':1E-18,              #   Minimum capture cross section for hole for defect 2
                'S_min_2_n':1E-18,          # Minimum capcture cross section for electron for defect 2
                'S_max_2_p':1E-12,              #   Maximum capture cross section for hole for defect 2
                'S_max_2_n':1E-12,          # Maximum capture cross section for electron for defect 2
                'Nt':1E12,                  #   Defect density
                'check_auger':True,        #   Check wether to resample if lifetime is auger-limited
                'noise':'',     #   Enable noiseparam
                'noiseparam':0,     #   Adds noise proportional to the log of Delta n
        },
        'generateSingle':{
                'type': 'p',                #   Wafer doping type
                'Et':-0.3,             #   Defect energy level
                'Sn':1E-14,              #   Electron capture cross section
                'Sp':1E-15,              #   Hole capture cross section
                'Nt':1E12,                  #   Defect density
                'CMn':'Radiative',    #   Capture mode for Sn
                'CMp':'Radiative',    #   Capture mode for Sp
                'CPn':None,    #   Capture mode parameters for Sn
                'CPp':None,    #   Capture mode parameters for Sp
                'name': None,   #   Defect name
                'noise':'',     #   Enable noiseparam
                'noiseparam':0,     #   Adds noise proportional to the log of Delta n
        }
    }

    #****   general methods     ****#
    def __init__(self,SaveDir,**Parameters):
        "Initialize object with passed or default parameters"

        #   Check applicability of method
        if not os.path.exists(SaveDir): raise ValueError('%s does not exists'%(SaveDir))

        #   Create directory for computation on this dataset
        self.pathDic = {
            'savedir':      SaveDir,
            'figures':      SaveDir+"\\figures\\",
            'objects':      SaveDir+"\\objects\\",
            'traces':       SaveDir+"\\traces\\",
            'outputs':      SaveDir+"\\outputs\\",
        }
        for key, value in self.pathDic.items():
                if key in ['figures','objects','traces','outputs']:
                    if not os.path.exists(value):   os.makedirs(value)

        #   define hyper parameters for experiment
        self.param = DPML.DefaultParameters.copy()
        self.param.update(Parameters)

    #****   simulation methods     ****#
    def generateDB(N, TEMP_RANGE = [200,250,300,350,400], DOP_RANGE = [1e15]*5, DN_RANGE= np.logspace(13,17,10), Parameters=None):
        "Generate defect database of size N, from object parameters. Each database generated will have a separate id."
        #   Update parameters
        param = DPML.DefaultParameters['generateDB'].copy()
        param.update(Parameters)
        # Generate Random defect database
        defectDB=Defect.randomDB(
            N=N,
            Et_min = param['Et_min'],
            Et_max = param['Et_max'],
            S_min_n = param['S_min_n'],
            S_min_p = param['S_min_p'],
            S_max_n = param['S_max_n'],
            S_max_p = param['S_max_p'],
            Nt = param['Nt'],
            CMn_tab = param['CMn_tab'],
            CMp_tab = param['CMp_tab'],
            Force_same_CM=param['Force_same_CM'],
        )

        # Calculate cell level characteristic for temperature,doping
        cref = Cell(T=300,Ndop=1E15,type=param['type'])
        cellDB = [cref.changeT(T).changeNdop(Ndop) for (T,Ndop) in zip(TEMP_RANGE,DOP_RANGE)]

        # Calculate lifetime on database and check for Auger limit
        columns_name = ["Name","Et_eV","Sn_cm2","Sp_cm2",'k','logSn','logSp','logk','bandgap','CMn','CPn','CMp','CPp']
        ltsDB=[]
        ltsDic=[]
        firstPass = True
        counter = 0
        for d in defectDB:
            # print which defect we are up to
            counter = counter + 1
            print(counter)
            bandgap = 1 if d.Et>0 else 0
            col = [d.name,d.Et,d.Sn,d.Sp,d.k, np.log10(d.Sn),np.log10(d.Sp),np.log10(d.k),bandgap,d.CMn,d.CPn,d.CMp,d.CPp]
            Sn0 = d.Sn
            Sp0 = d.Sp
            skipDefect = False
            curves=[]
            for c in cellDB:
                if firstPass:
                    for dn in DN_RANGE: columns_name.append("%sK_%scm-3_ %scm-3" % (c.T,c.Ndop,dn))
                    skipDefect = True
                    continue
                if skipDefect: continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    d_loc=d.copy()

                #   Adjust Sn, Sp from Capture model
                # if d.CMn == 'Multiphonon emission': d.Sn=Sn0*np.exp(-d.CPn/(Cell.kb*c.T))
                # if d.CMn == 'Cascade': d.Sn=Sn0*np.power(c.T,-d.CPn)
                # if d.CMp == 'Multiphonon emission': d.Sp=Sp0*np.exp(-d.CPp/(Cell.kb*c.T))
                # if d.CMp == 'Cascade': d.Sp=Sp0*np.power(c.T,-d.CPp)
                if d_loc.CMn == 'Multiphonon emission': d_loc.Sn=float(Decimal(Sn0)*np.exp(Decimal(-d_loc.CPn/(Cell.kb*c.T))))
                if d_loc.CMn == 'Cascade': d_loc.Sn=float(Decimal(Sn0)*Decimal(np.power(float(c.T),-d_loc.CPn)))
                if d_loc.CMp == 'Multiphonon emission': d_loc.Sp=float(Decimal(Sp0)*np.exp(Decimal(-d_loc.CPp/(Cell.kb*c.T))))
                if d_loc.CMp == 'Cascade': d_loc.Sp=float(Decimal(Sp0)*Decimal(np.power(float(c.T),-d_loc.CPp)))
                # s = LTS(c,d,DN_RANGE,noise="", noiseparam=0)
                s = LTS(c,d_loc,DN_RANGE,noise=param['noise'], noiseparam=param['noiseparam'])

                if param['check_auger']:  #if break auger limit, discard defect
                    breakAuger,_ = s.checkAuger()
                    if breakAuger: skipDefect=True
                if skipDefect: continue
                curves.append(s)
                for t,dn in zip(s.tauSRH_noise,s.dnrange):    # ADD NOISE HERE
                    col.append(t)
            firstPass = False
            if not skipDefect:
                ltsDB.append(col)
                ltsDic.append(curves)
            if skipDefect:  # if we skipped a defect, add a new random one to have n_defects in the database at the end
                newDefect = Defect.randomDB(
                        N=1,
                        Et_min = param['Et_min'],
                        Et_max = param['Et_max'],
                        S_min_n = param['S_min_n'],
                        S_min_p = param['S_min_p'],
                        S_max_n = param['S_max_n'],
                        S_max_p = param['S_max_p'],
                        Nt = param['Nt'],
                        CMn_tab = [d.CMn],
                        CMp_tab = [d.CMp],
                        Force_same_CM=False,
                    )[0]
                newDefect.name = d.name
                defectDB.append(newDefect)
        ltsDF = pd.DataFrame(ltsDB)
        ltsDF.columns = columns_name
        # if self.parameters['save']:
        #     SaveObj(ltsDF,self.pathDic['objects'],'ltsDF_ID'+ltsID+"_"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        #     self.updateLogbook('lifetime_database_saved_ID'+ltsID)
        return ltsDF,ltsDic
    def generateDB_multi(N, TEMP_RANGE = [200,250,300,350,400], DOP_RANGE = [1e15]*5, DN_RANGE= np.logspace(13,17,10),Parameters=None):
        "Generate multiple one-level defect database of size N, from object parameters. Each database generated will have a separate id."
        #   Update parameters
        param = DPML.DefaultParameters['generateDB_multi'].copy()
        param.update(Parameters)
        # Generate Random defect database
        defectDB_1=Defect.randomDB(
            N=N,
            Et_min = param['Et_min_1'],
            Et_max = param['Et_max_1'],
            S_min_n = param['S_min_1_n'],
            S_min_p = param['S_min_1_p'],
            S_max_n = param['S_max_1_n'],
            S_max_p = param['S_max_1_p'],
            Nt = param['Nt_1'],
        )
        defectDB_2=Defect.randomDB(
            N=N,
            Et_min = param['Et_min_2'],
            Et_max = param['Et_max_2'],
            S_min_n = param['S_min_2_n'],
            S_min_p = param['S_min_2_p'],
            S_max_n = param['S_max_2_n'],
            S_max_p = param['S_max_2_p'],
            Nt = param['Nt_2'],
        )

        # Calculate cell level characteristic for temperature,doping
        cref = Cell(T=300,Ndop=1E15,type=param['type'])
        cellDB = [cref.changeT(T).changeNdop(Ndop) for (T,Ndop) in zip(TEMP_RANGE,DOP_RANGE)]

        # Calculate lifetime on database and check for Auger limit
        columns_name = [
            "Name",
            "Et_eV_1",
            "Sn_cm2_1",
            "Sp_cm2_1",
            'k_1',
            'logSn_1',
            'logSp_1',
            'logk_1',
            'bandgap_1',
            "Et_eV_2",
            "Sn_cm2_2",
            "Sp_cm2_2",
            'k_2',
            'logSn_2',
            'logSp_2',
            'logk_2',
            'bandgap_2',
            ]
        ltsDB=[]
        firstPass = True
        counter = 0
        for dA,dB in zip(defectDB_1,defectDB_2):
            # print which defect we are up to
            counter = counter + 1
            print(counter)
            #   Assign defect 1 and 2
            if dA.Et>dB.Et:
                d1=dA
                d2=dB
            else:
                # d1=dB
                # d2=dA
                d1=dA
                d2=dB
            bg_1 = 1 if d1.Et>0 else 0
            bg_2 = 1 if d2.Et>0 else 0
            col = [
                d1.name,
                d1.Et,
                d1.Sn,
                d1.Sp,
                d1.k,
                np.log10(d1.Sn),
                np.log10(d1.Sp),
                np.log10(d1.k),
                bg_1,
                d2.Et,
                d2.Sn,
                d2.Sp,
                d2.k,
                np.log10(d2.Sn),
                np.log10(d2.Sp),
                np.log10(d2.k),
                bg_2,
            ]
            skipDefect = False
            for c in cellDB:
                if firstPass:
                    for dn in DN_RANGE: columns_name.append("%sK_%scm-3_ %scm-3" % (c.T,c.Ndop,dn))
                    skipDefect = True
                    continue
                if skipDefect: continue
                s1 = LTS(c,d1,DN_RANGE,noise="", noiseparam=0)
                s2 = LTS(c,d2,DN_RANGE,noise="", noiseparam=0)
                if param['check_auger']:  #if break auger limit, discard defect
                    breakAuger1,_ = s1.checkAuger()
                    breakAuger2,_ = s2.checkAuger()
                    if breakAuger1: skipDefect=True
                    if breakAuger2: skipDefect=True
                if skipDefect: continue
                for t1,t2 in zip(s1.tauSRH,s2.tauSRH):    # ADD NOISE HERE
                    col.append(1/(1/t1+1/t2))
            firstPass = False
            if not skipDefect: ltsDB.append(col)
            if skipDefect:  # if we skipped a defect, add a new random one to have n_defects in the database at the end
                new_d1 = Defect.randomDB(
                        N=1,
                        Et_min = param['Et_min_1'],
                        Et_max = param['Et_max_1'],
                        S_min_n = param['S_min_1_n'],
                        S_min_p = param['S_min_1_p'],
                        S_max_n = param['S_max_1_n'],
                        S_max_p = param['S_max_1_p'],
                        Nt = param['Nt_1'],
                    )[0]
                new_d1.name = d1.name
                defectDB_1.append(new_d1)
                new_d2 = Defect.randomDB(
                        N=1,
                        Et_min = param['Et_min_2'],
                        Et_max = param['Et_max_2'],
                        S_min_n = param['S_min_2_n'],
                        S_min_p = param['S_min_2_p'],
                        S_max_n = param['S_max_2_n'],
                        S_max_p = param['S_max_2_p'],
                        Nt = param['Nt_2'],
                    )[0]
                new_d2.name = d2.name
                defectDB_2.append(new_d2)
        ltsDF = pd.DataFrame(ltsDB)
        ltsDF.columns = columns_name
        # if self.parameters['save']:
        #     SaveObj(ltsDF,self.pathDic['objects'],'ltsDF_ID'+ltsID+"_"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        #     self.updateLogbook('lifetime_database_saved_ID'+ltsID)
        return ltsDF

    def generateDB_sah(N, TEMP_RANGE = [200,250,300,350,400], DOP_RANGE = [1e15]*5, DN_RANGE= np.logspace(13,17,10),Parameters=None):
        "Generate two-level defect database of size N, from object parameters. Each database generated will have a separate id."
        # print('a')
        #   Update parameters
        param = DPML.DefaultParameters['generateDB_sah'].copy()
        param.update(Parameters)
        # Generate Random defect database
        defectDB_1=Defect.randomDB(
            N=N,
            Et_min = param['Et_min_1'],
            Et_max = param['Et_max_1'],
            S_min_n = param['S_min_1_n'],
            S_min_p = param['S_min_1_p'],
            S_max_n = param['S_max_1_n'],
            S_max_p = param['S_max_1_p'],
            Nt = param['Nt'],
        )
        # sanity check:
        # print(param['S_max_1_n'])
        # print(param['S_min_1_n'])
        # print(param['S_max_2_n'])
        # print(param['S_min_2_n'])
        defectDB_2=Defect.randomDB(
            N=N,
            Et_min = param['Et_min_2'],
            Et_max = param['Et_max_2'],
            S_min_n = param['S_min_2_n'],
            S_min_p = param['S_min_2_p'],
            S_max_n = param['S_max_2_n'],
            S_max_p = param['S_max_2_p'],
            Nt = param['Nt'],
        )
        # sanity check:
        # print(param['S_max_1_n'])
        # print(param['S_min_1_n'])
        # print(param['S_max_2_n'])
        # print(param['S_min_2_n'])
        # print(param['S_max_2_p'])
        # sanity check:
        # print(defectDB_1)
        # print(defectDB_2)
        # Calculate cell level characteristic for temperature,doping
        cref = Cell(T=300,Ndop=1E15,type=param['type'])
        cellDB = [cref.changeT(T).changeNdop(Ndop) for (T,Ndop) in zip(TEMP_RANGE,DOP_RANGE)]

        # Calculate lifetime on database and check for Auger limit
        columns_name = [
            "Name",
            "Et_eV_1",
            "Sn_cm2_1",
            "Sp_cm2_1",
            'k_1',
            'logSn_1',
            'logSp_1',
            'logk_1',
            'bandgap_1',
            "Et_eV_2",
            "Sn_cm2_2",
            "Sp_cm2_2",
            'k_2',
            'logSn_2',
            'logSp_2',
            'logk_2',
            'bandgap_2',
            ]
        ltsDB=[]
        firstPass = True
        counter = 0
        for dA,dB in zip(defectDB_1,defectDB_2):
            counter = counter + 1
            print(counter)
            #   Assign defect 1 and 2
            # make sure Et1 is always larger than Et2, otherwise rename them to exchange the name.
            if dA.Et>dB.Et:
                d1=dA
                d2=dB
                bg_1 = 1 if d1.Et>0 else 0
                bg_2 = 1 if d2.Et>0 else 0
                col = [
                    d1.name,
                    d1.Et,
                    d1.Sn,
                    d1.Sp,
                    d1.k,
                    np.log10(d1.Sn),
                    np.log10(d1.Sp),
                    np.log10(d1.k),
                    bg_1,
                    d2.Et,
                    d2.Sn,
                    d2.Sp,
                    d2.k,
                    np.log10(d2.Sn),
                    np.log10(d2.Sp),
                    np.log10(d2.k),
                    bg_2,
                ]
                skipDefect = False
            else:
                # here we do not skip the case when Et2>Et1 anymore.
                d1=dA
                d2=dB
                bg_1 = 1 if d1.Et>0 else 0
                bg_2 = 1 if d2.Et>0 else 0
                col = [
                    d1.name,
                    d1.Et,
                    d1.Sn,
                    d1.Sp,
                    d1.k,
                    np.log10(d1.Sn),
                    np.log10(d1.Sp),
                    np.log10(d1.k),
                    bg_1,
                    d2.Et,
                    d2.Sn,
                    d2.Sp,
                    d2.k,
                    np.log10(d2.Sn),
                    np.log10(d2.Sp),
                    np.log10(d2.k),
                    bg_2,
                ]
                skipDefect = False

            for c in cellDB:
                if firstPass:
                    for dn in DN_RANGE: columns_name.append("%sK_%scm-3_ %scm-3" % (c.T,c.Ndop,dn)) # define a new column, name it with doping and T.
                    skipDefect = True
                    continue
                if skipDefect: continue
                s = Sah(c,d1,d2,DN_RANGE)
                if param['check_auger']:  #if break auger limit, discard defect
                    breakAuger,_ = s.checkAuger()
                    if breakAuger: skipDefect=True
                if skipDefect: continue
                for t in s.tauSah:    # ADD NOISE HERE
                    col.append(t)
            firstPass = False
            if not skipDefect: ltsDB.append(col)
            if skipDefect:  # if we skipped a defect, add a new random one to have n_defects in the database at the end
                new_d1 = Defect.randomDB(
                        N=1,
                        Et_min = param['Et_min_1'],
                        Et_max = param['Et_max_1'],
                        S_min_n = param['S_min_1_n'],
                        S_min_p = param['S_min_1_p'],
                        S_max_n = param['S_max_1_n'],
                        S_max_p = param['S_max_1_p'],
                        Nt = param['Nt'],
                    )[0]
                new_d1.name = d1.name
                defectDB_1.append(new_d1)
                new_d2 = Defect.randomDB(
                        N=1,
                        Et_min = param['Et_min_2'],
                        Et_max = param['Et_max_2'],
                        S_min_n = param['S_min_2_n'],
                        S_min_p = param['S_min_2_p'],
                        S_max_n = param['S_max_2_n'],
                        S_max_p = param['S_max_2_p'],
                        Nt = param['Nt'],
                    )[0]
                new_d2.name = d2.name
                defectDB_2.append(new_d2)
        ltsDF = pd.DataFrame(ltsDB)
        ltsDF.columns = columns_name
        # if self.parameters['save']:
        #     SaveObj(ltsDF,self.pathDic['objects'],'ltsDF_ID'+ltsID+"_"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        #     self.updateLogbook('lifetime_database_saved_ID'+ltsID)
        return ltsDF

    def generateDB_sah_both_np(N, TEMP_RANGE = [200,250,300,350,400], DOP_RANGE = [1e15]*5, DN_RANGE= np.logspace(13,17,10),Parameters=None):
        "Generate two-level defect database of size N, from object parameters. Each database generated will have a separate id."
        # print('a')
        #   Update parameters
        param = DPML.DefaultParameters['generateDB_sah'].copy()
        param.update(Parameters)
        # Generate Random defect database
        defectDB_1=Defect.randomDB(
            N=N,
            Et_min = param['Et_min_1'],
            Et_max = param['Et_max_1'],
            S_min_n = param['S_min_1_n'],
            S_min_p = param['S_min_1_p'],
            S_max_n = param['S_max_1_n'],
            S_max_p = param['S_max_1_p'],
            Nt = param['Nt'],
        )
        # sanity check:
        # print(param['S_max_1_n'])
        # print(param['S_min_1_n'])
        # print(param['S_max_2_n'])
        # print(param['S_min_2_n'])
        defectDB_2=Defect.randomDB(
            N=N,
            Et_min = param['Et_min_2'],
            Et_max = param['Et_max_2'],
            S_min_n = param['S_min_2_n'],
            S_min_p = param['S_min_2_p'],
            S_max_n = param['S_max_2_n'],
            S_max_p = param['S_max_2_p'],
            Nt = param['Nt'],
        )
        # sanity check:
        # print(param['S_max_1_n'])
        # print(param['S_min_1_n'])
        # print(param['S_max_2_n'])
        # print(param['S_min_2_n'])
        # print(param['S_max_2_p'])
        # sanity check:
        # print(defectDB_1)
        # print(defectDB_2)
        # Calculate cell level characteristic for temperature,doping

        cref_n = Cell(T=300,Ndop=1E15,type='n')
        cellDB_n = [cref_n.changeT(T).changeNdop(Ndop) for (T,Ndop) in zip(TEMP_RANGE,DOP_RANGE)]

        # Calculate lifetime on database and check for Auger limit
        columns_name = [
            "Name",
            "Et_eV_1",
            "Sn_cm2_1",
            "Sp_cm2_1",
            'k_1',
            'logSn_1',
            'logSp_1',
            'logk_1',
            'bandgap_1',
            "Et_eV_2",
            "Sn_cm2_2",
            "Sp_cm2_2",
            'k_2',
            'logSn_2',
            'logSp_2',
            'logk_2',
            'bandgap_2',
            ]
        ltsDB=[] # it is a list
        firstPass = True
        # for each defect:
        counter = 0
        for dA,dB in zip(defectDB_1,defectDB_2):
            counter  = counter + 1
            print(counter)
            #   Assign defect 1 and 2
            # make sure Et1 is always larger than Et2, otherwise rename them to exchange the name.
            if dA.Et>dB.Et:
                d1=dA
                d2=dB
                bg_1 = 1 if d1.Et>0 else 0
                bg_2 = 1 if d2.Et>0 else 0
                col = [
                    d1.name,
                    d1.Et,
                    d1.Sn,
                    d1.Sp,
                    d1.k,
                    np.log10(d1.Sn),
                    np.log10(d1.Sp),
                    np.log10(d1.k),
                    bg_1,
                    d2.Et,
                    d2.Sn,
                    d2.Sp,
                    d2.k,
                    np.log10(d2.Sn),
                    np.log10(d2.Sp),
                    np.log10(d2.k),
                    bg_2,
                ]
                skipDefect = False
            else:
                # here we do not skip the case when Et2>Et1 anymore.
                d1=dA
                d2=dB
                bg_1 = 1 if d1.Et>0 else 0
                bg_2 = 1 if d2.Et>0 else 0
                col = [
                    d1.name,
                    d1.Et,
                    d1.Sn,
                    d1.Sp,
                    d1.k,
                    np.log10(d1.Sn),
                    np.log10(d1.Sp),
                    np.log10(d1.k),
                    bg_1,
                    d2.Et,
                    d2.Sn,
                    d2.Sp,
                    d2.k,
                    np.log10(d2.Sn),
                    np.log10(d2.Sp),
                    np.log10(d2.k),
                    bg_2,
                ]
                skipDefect = False

            for c in cellDB_n:
                if firstPass:
                    for dn in DN_RANGE: columns_name.append("%sK_%scm-3_ %scm-3_n" % (c.T,c.Ndop,dn)) # define a new column, name it with doping and T.
                    skipDefect = True
                    continue
                if skipDefect: continue
                s = Sah(c,d1,d2,DN_RANGE)
                if param['check_auger']:  #if break auger limit, discard defect
                    breakAuger,_ = s.checkAuger()
                    if breakAuger: skipDefect=True
                if skipDefect: continue
                for t in s.tauSah:    # ADD NOISE HERE
                    col.append(t)
            # firstPass = False
            # if not skipDefect: ltsDB.append(col) # append the column.
            # print(len(columns_name))

            # repeat the process for p type:
            cref_p = Cell(T=300,Ndop=1E15,type='p')
            cellDB_p = [cref_p.changeT(T).changeNdop(Ndop) for (T,Ndop) in zip(TEMP_RANGE,DOP_RANGE)]
            for c in cellDB_p:
                if firstPass:
                    for dn in DN_RANGE: columns_name.append("%sK_%scm-3_ %scm-3_p" % (c.T,c.Ndop,dn)) # define a new column, name it with doping and T.
                    skipDefect = True
                    continue
                if skipDefect: continue
                s = Sah(c,d1,d2,DN_RANGE)
                if param['check_auger']:  #if break auger limit, discard defect
                    breakAuger,_ = s.checkAuger()
                    if breakAuger: skipDefect=True
                if skipDefect: continue
                for t in s.tauSah:    # ADD NOISE HERE
                    col.append(t)
            firstPass = False
            if not skipDefect: ltsDB.append(col) # append the column.
            # print(len(columns_name))

            if skipDefect:  # if we skipped a defect, add a new random one to have n_defects in the database at the end
                new_d1 = Defect.randomDB(
                        N=1,
                        Et_min = param['Et_min_1'],
                        Et_max = param['Et_max_1'],
                        S_min_n = param['S_min_1_n'],
                        S_min_p = param['S_min_1_p'],
                        S_max_n = param['S_max_1_n'],
                        S_max_p = param['S_max_1_p'],
                        Nt = param['Nt'],
                    )[0]
                new_d1.name = d1.name
                defectDB_1.append(new_d1)
                new_d2 = Defect.randomDB(
                        N=1,
                        Et_min = param['Et_min_2'],
                        Et_max = param['Et_max_2'],
                        S_min_n = param['S_min_2_n'],
                        S_min_p = param['S_min_2_p'],
                        S_max_n = param['S_max_2_n'],
                        S_max_p = param['S_max_2_p'],
                        Nt = param['Nt'],
                    )[0]
                new_d2.name = d2.name
                defectDB_2.append(new_d2)

        ltsDF = pd.DataFrame(ltsDB)
        print(ltsDF.shape)
        ltsDF.columns = columns_name
        # if self.parameters['save']:
        #     SaveObj(ltsDF,self.pathDic['objects'],'ltsDF_ID'+ltsID+"_"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        #     self.updateLogbook('lifetime_database_saved_ID'+ltsID)
        return ltsDF


    def generateSingle(TEMP_RANGE = [200,250,300,350,400], DOP_RANGE = [1e15]*5, DN_RANGE= np.logspace(13,17,10),Parameters=None):
        """Generate and return a single defect object"""
        #   Update parameters
        param = DPML.DefaultParameters['generateSingle'].copy()
        param.update(Parameters)
        ltsDic={}
        # Generate Random defect database
        d=Defect(
            Et = param['Et'],
            Sn = param['Sn'],
            Sp = param['Sp'],
            Nt = param['Nt'],
            Capture_mode_n=param['CMn'],
            Capture_mode_p=param['CMp'],
            Capture_param_n=param['CPn'],
            Capture_param_p=param['CPp'],
            name=param['name'],
        )
        cref = Cell(T=300,Ndop=1E15,type=param['type'])
        cellDB = [cref.changeT(T).changeNdop(Ndop) for (T,Ndop) in zip(TEMP_RANGE,DOP_RANGE)]
        # Calculate lifetime on database and check for Auger limit
        ltsDic['defect']=d
        ltsDic['cellDB']=cellDB
        Sn0 = d.Sn
        Sp0 = d.Sp
        sDB=[]
        for c in cellDB:
            d_loc=d.copy()
            #   Adjust Sn, Sp from Capture model
            if d_loc.CMn == 'Multiphonon emission': d_loc.Sn=float(Decimal(Sn0)*np.exp(Decimal(-d_loc.CPn/(Cell.kb*c.T))))
            if d_loc.CMn == 'Cascade': d_loc.Sn=float(Decimal(Sn0)*Decimal(np.power(float(c.T),-d_loc.CPn)))
            if d_loc.CMp == 'Multiphonon emission': d_loc.Sp=float(Decimal(Sp0)*np.exp(Decimal(-d_loc.CPp/(Cell.kb*c.T))))
            if d_loc.CMp == 'Cascade': d_loc.Sp=float(Decimal(Sp0)*Decimal(np.power(float(c.T),-d_loc.CPp)))
            # if d_loc.CMn == 'Multiphonon emission': d_loc.Sn=Sn0*np.exp(-d_loc.CPn/(Cell.kb*c.T))
            # if d_loc.CMn == 'Cascade': d_loc.Sn=Sn0*np.power(float(c.T),-d_loc.CPn)
            # if d_loc.CMp == 'Multiphonon emission': d_loc.Sp=Sp0*np.exp(-d_loc.CPp/(Cell.kb*c.T))
            # if d_loc.CMp == 'Cascade': d_loc.Sp=Sp0*np.power(float(c.T),-d_loc.CPp)
            s = LTS(c,d_loc,DN_RANGE,noise=param['noise'], noiseparam=param['noiseparam'])
            sDB.append(s)
        ltsDic['sDB']=sDB
        return ltsDic
