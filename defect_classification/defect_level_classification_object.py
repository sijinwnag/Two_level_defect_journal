# %% Imports
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score, mean_absolute_error, confusion_matrix, f1_score, accuracy_score, classification_report
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
from joblib import dump, load


# %% define the object


class defect_classifier():
    '''
    This object is related to classifier whether the defect is one or two-level based on testing sets

    train_test_same: a boolean input, if it is true, then the training and testing are on the same csv file.
    '''

    def __init__(self):
        # define the training path, it will be updatable
        self.training_path = r"C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\yoann_code_new\Savedir_example\outputs\outputs\2023-01-20-21-03-20_advanced example - multi_level_L_datasetID_0.csv"

        # define the testing path
        self.test_path = r"C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\yoann_code_new\Savedir_example\outputs\outputs\2023-01-20-21-43-28_advanced example - multi_level_L_datasetID_0.csv"

        # define the classification model: we already know to use neural network
        self.model = MLPClassifier((100, 100),alpha=0.001, activation = 'relu',verbose=0,learning_rate='adaptive')

        # define the scaler to use
        self.scaler = MinMaxScaler()
        
        # define the parameter grid
        self.param_grid = {'hidden_layer_size':[(100, 100), (100, 200), (500, 500), (100, 100, 100)]}
        
    
    def train_export_model(self):
        '''
        Input: training_path: the path of the training data
        Output: exrported scaler and model
        '''
        # load the training data
        self.training_data = pd.read_csv(self.training_path)

        # define the ML input
        # create a list to select X columns: if the column string contains cm, then identify it as X.
        select_X_list = []
        for string in self.training_data.columns.tolist():
            if string[0].isdigit():
                select_X_list.append(string)
        X = self.training_data[select_X_list]

        # take the log 10 of the training data (lifetime data)
        X = np.log10(X)

        # define y for ML output
        y = self.training_data['Label']

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        # apply the scaler
        scaler = self.scaler
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # train the classifier
        print('training in progress')
        self.model.fit(X_train_scaled, y_train)

        # test with test set
        y_pred = self.model.predict(X_test_scaled)

        # classification report
        print(classification_report(y_test, y_pred))

        # export the scaler and model
        dump(scaler, 'scaler_for_defect_classification.joblib')
        dump(self.model, 'model_defect_classification.joblib')


    def train_export_transfer_learning(self):
        '''
        Input: training_path: the path of the training data
        Output: exrported scaler and model
        '''
    

    def train_Gridsearch(self):
        '''
        Input: training path
        Output : exported scaler and model
        '''
        # create the grid search object
        grid_search = GridSearchCV(self.model, param_grid = self.param_grid, cv=5)

        # apply the data pre processing
        # load the training data
        self.training_data = pd.read_csv(self.training_path)

        # define the ML input
        # create a list to select X columns: if the column string contains cm, then identify it as X.
        select_X_list = []
        for string in self.training_data.columns.tolist():
            if string[0].isdigit():
                select_X_list.append(string)
        X = self.training_data[select_X_list]

        # take the log 10 of the training data (lifetime data)
        X = np.log10(X)

        # define y for ML output
        y = self.training_data['Label']

        # train test split
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        # apply the scaler
        scaler = self.scaler
        scaler.fit(X)
        X_scaled = scaler.transform(X)
    
        # train the grid search object
        grid_search.fit(X_scaled, y)

        # print hte best parameters and the best score
        print('The best parameters: ' + str(grid_search.best_params_))
        print('Best score: ', str(grid_search.best_score_))

        # store the best parameters into the object
        self.best_params = grid_search.best_params_



    def test_model(self):
        '''
        Input:
            trained_model_path
            trained_scaler_path
            test_data_path

        Output:
            a classification report regarding the testing 
        '''
        # define the trained model, trained scaler and test data path
        trained_model_path = self.trained_model_path
        trained_scaler_path = self.trained_scaler_path
        test_data_path = self.test_data_path

        # load the ML model
        test_model = load(trained_model_path)

        # load the scaler object
        scaler = load(trained_scaler_path)

        # load the test data
        test_df = pd.read_csv(test_data_path)

        # extract the X
        # create a list to select X columns: if the column string contains cm, then identify it as X.
        select_X_list = []
        for string in test_df.columns.tolist():
            if string[0].isdigit():
                select_X_list.append(string)
        X = test_df[select_X_list]
        # take the log 10 of the training data (lifetime data)
        X = np.log10(X)
        # go through the scaler
        X_scaled = scaler.transform(X)

        # make the prediction from loaded model
        y_test = test_df['Label']
        y_pred = test_model.predict(X_scaled)

        # make a report
        print(classification_report(y_test, y_pred))