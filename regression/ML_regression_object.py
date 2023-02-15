# %% imports
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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
import joblib
import time

# %% define the object

class ML_regression():
    '''
    This object aim to use ML regression to predict the 6 parameter for two level defect based on Sah-Shockley Equation
    '''

    def __init__(self):
        # define the path for the training dataset
        self.training_path = r'G:\study\thesis_data_storage\unordered\set10\p\8k\2022-11-24-11-05-49_advanced example - multi_level_L_datasetID_0.csv'

        # define the ML model
        self.model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1)

        # defien the scaler
        self.scaler = MinMaxScaler()

        # define the path where the trained model is stored
        self.model_path = r'G:\study\thesis_data_storage\journal\regression\trained_model\Set10\trained_ML_model\Et_eV_1Completed_model.joblib'

        # define the path for test dataset
        self.testing_path = r'G:\study\thesis_data_storage\unordered\set10\p\8k\2022-11-24-11-05-49_advanced example - multi_level_L_datasetID_0.csv'

        # define the path for trained scaler
        self.scaler_path = r'G:\study\thesis_data_storage\journal\regression\trained_model\Set10\trained_ML_model\scaler_forEt_eV_1Completed_model.joblib'

        # define the y for regression
        self.y_str = 'Et_eV_1'
    

    def load_test_model(self):
        '''
        Load the trained model and apply on testset
        '''
        # start recording the time
        start_time = time.time()

        # load the test data
        test_data = pd.read_csv(self.testing_path)

        # load the trained model
        loaded_model = joblib.load(self.model_path)

        # load the trained scaler
        loaded_scaler = joblib.load(self.scaler_path)

        # get X and y for pre-processor
        X, y = self.pre_processor(test_data)

        # apply scaler
        X_scaled = loaded_scaler.transform(X)

        # apply the ML model
        y_pred = loaded_model.predict(X_scaled)

        # recording the ending time
        end_time = time.time()

        # record the computational time
        dt = end_time - start_time
        self.computation_time = dt
        print('Computational time is ' + str(dt))

        # plot the real vs prediction
        plt.figure()
        plt.scatter(y, y_pred)
        plt.show()

        # compute the confusion matrix
        r2 = r2_score(y, y_pred)
        print('R2 score is ' + str(r2))
        self.r2 = r2

        mae = mean_absolute_error(y, y_pred)
        print('Mean Absolute Error: ' + str(mae))
        self.mae = mae

        mse = mean_squared_error(y, y_pred)
        print('Mean Square Error: ' + str(mse))
        self.mse = mse

        # store the output into the object
        self.y_pred = y_pred
        self.y_test = y


    def pre_processor(self, csv_data):
        '''
        Input: a csv file from lifetime simulator
        Output: X lifetime array, y target value
        '''
        # extract the lifetime
        # define the ML input
        # create a list to select X columns: if the column string contains cm, then identify it as X.
        select_X_list = []
        for string in csv_data.columns.tolist():
            if string[0].isdigit():
                select_X_list.append(string)
        X = csv_data[select_X_list]

        # take the log 10 of the training data (lifetime data)
        X = np.log10(X)

        # define y for ML output
        y = csv_data[self.y_str]

        return X, y


    def train_test_model(self):
        '''
        This function aims to train and test the model
        '''
        # load the training data
        training_data = pd.read_csv(self.training_path)

        # extract x and y
        X, y = self.pre_processor(training_data)

        # train_test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        # define the scaler
        scaler = self.scaler
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # store the scaler into object
        self.trained_scaler = scaler

        # define the model
        model = self.model
        model.fit(X_train_scaled, y_train)
        # store the model into object
        self.trained_model = model

        # make the prediction
        y_pred = model.predict(X_test_scaled)

        # plot the real vs prediction
        plt.figure()
        plt.scatter(y_test, y_pred, alpha = self.transparency_calculator(np.shape(y_test)[0]))
        plt.show()

        # compute the confusion matrix
        r2 = r2_score(y_test, y_pred)
        print('R2 score is ' + str(r2))
        self.r2 = r2

        mae = mean_absolute_error(y_test, y_pred)
        print('Mean Absolute Error: ' + str(mae))
        self.mae = mae

        mse = mean_squared_error(y_test, y_pred)
        print('Mean Square Error: ' + str(mse))
        self.mse = mse

    
    def transparency_calculator(self, datasize):
        '''
        This function will calcualte a suitable data transparency given the datasize for a scatter plot.

        input: datasize: an integer.
        '''
        if datasize>800:
            alpha = 800/datasize*0.5
        else:
            alpha = 0.5
        return alpha


    def export_model_scaler(self):
        '''
        This function export the trained model and scaler
        '''
        joblib.dump(self.trained_model, 'trained_model_' + str(self.y_str) + '.joblib')
        joblib.dump(self.trained_scaler, 'trained_scaler_' + str(self.y_str) + '.joblib')