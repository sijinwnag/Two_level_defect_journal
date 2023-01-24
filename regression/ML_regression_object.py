# %% imports
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score, mean_absolute_error, confusion_matrix, f1_score, accuracy_score
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
import sys
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from playsound import playsound
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.multioutput import RegressorChain
from semiconductor.recombination import SRH
import scipy.constants as sc
from datetime import datetime
import smtplib
from email.message import EmailMessage
import os
from sklearn.inspection import permutation_importance
import sympy as sym

# %% define the object

# class ML_regression():
#     '''
#     This object aim to use ML regression to predict the 6 parameter for two level defect based on Sah-Shockley Equation
#     '''

#     def __init__(self):
#         # define the path for the training dataset
#         self.training_path = r'C:\Users\sijin wang\Documents\GitHub\Two_level_defect_journal\simulation\simulated_data\outputs\2023-01-24-14-42-33_advanced example - multi_level_L_datasetID_0.csv'

#         # define the ML model
#         self.model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1)

    
#     def load_