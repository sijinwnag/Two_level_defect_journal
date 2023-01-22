# %%-- Imports
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
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
# uncomment the below line for dell laptop only
from playsound import playsound
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.multioutput import RegressorChain
from semiconductor.recombination import SRH
import scipy.constants as sc
from datetime import datetime
import smtplib
from email.message import EmailMessage
import os
import sys
# import the function file from another folder:
# use this line if on hp laptop:
# sys.path.append(r'C:\Users\budac\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2')
# use this line if on dell laptop
# sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2')
# use this line if on workstation
sys.path.append(r'C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2')
sys.path.append(r'C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\DPML')
sys.path.append(r'C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Savedir_example')
from MLobject_tlevel import *
# from dynamic_generation_regression import *
df1 = MyMLdata_2level(r'C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Savedir_example'), 'bandgap1',5)
# df1.data.head()
# df1.data['Label']
# %%-

# %%-- perform classification
df1.singletask='Label'
# try svc by itself
# X, y = df1.pre_processor()
f1scores, y_prediction_frame, y_test_frame = df1.classification_repeat()
df1.email_reminder()
# %%-

# %%-- learning curve.
# create an empty list to collect results.
f1_list = []
for fraction in np.logspace(-3, 0, 20):
    # update the object.
    df1 = MyMLdata_2level(r"C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\simulation_data\Etnonordered\one_vs_two_level_classification\80k\2022-09-14-10-03-50_advanced example - multi_level_L_datasetID_0.csv", 'bandgap1',5)
    # resample the data.
    df1.data = df1.data.sample(frac=fraction)
    # define the task.
    df1.singletask='Label'
    # perform the training and testing.
    # print('starting training iteration')
    f1scores, y_prediction_frame, y_test_frame = df1.classification_repeat()
    f1_list.append(f1scores)
# send an email to myself after done.
df1.email_reminder()
# %%-


# %%-- plot the results: data processing.
# saperate out the nn results.
f1_list2 = f1_list
f1_array = np.array(f1_list2)
f1_array_nn = f1_array[:, :, 1]
f1_array_nn = np.array(f1_array_nn)
# calcualte the average, medium, and std for each data size.
f1_av = np.average(f1_array_nn, axis=1)
f1_av
f1_med = np.median(f1_array_nn, axis=1)
f1_med
f1_std = np.std(f1_array_nn, axis=1)
# create the datasize axis:
datasize = np.logspace(-3, 0, 20)*160e3*0.9
# %%-


# %%--plot the av with std vs datasize.
plt.figure(facecolor='white')
plt.errorbar(datasize, f1_av, yerr = f1_std, ecolor='black')
plt.xlabel('training datasize')
plt.ylabel('F-1 score')
plt.title('Average F-1 score vs training datasize')
# make the datasize above which the F1 saturates.
plt.axvline(x=120000, color='red', label='axvline - full height', ls='--')
# export the image.
plt.savefig('Average F-1 score vs training datasize' + '.png')
plt.show()
# %%-


# %%--plot the median with std vs datasize.
plt.figure(facecolor='white')
plt.errorbar(datasize, f1_med, yerr = f1_std, ecolor='black')
plt.xlabel('training datasize')
plt.ylabel('F-1 score')
plt.title('Median F-1 score vs training datasize')
# make the datasize above which the F1 saturates.
plt.axvline(x=120000, color='red', label='axvline - full height', ls='--')
# export the image.
plt.savefig('Median F-1 score vs training datasize' + '.png')
plt.show()
# %%-
