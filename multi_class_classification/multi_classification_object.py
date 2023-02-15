# %%---import libraries:
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.colors
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


# %%

class multi_classifier():
    '''
    This object is related to the classification task that split dataset into 4 different sets
    '''

    def __init__(self):
        # define the ML model
        self.model = MLPClassifier(hidden_layer_sizes=(512, 512, 512), verbose=1)

        # define the scaler
        self.scaler = MinMaxScaler()

        # define the training path
        self.set11_path = r'G:\study\thesis_data_storage\unordered\set11\p\outputs\set11_80k.csv'
        self.set10_path = r'G:\study\thesis_data_storage\unordered\set10\p\80k\2022_10_25\2022-10-25-13-57-56_advanced example - multi_level_L_datasetID_0.csv'
        self.set01_path = r'G:\study\thesis_data_storage\unordered\set01\p\outputs\outputs\set01_80k.csv'
        self.set00_path = r'G:\study\thesis_data_storage\unordered\set00\p\set00_80k\outputs\set00_80k.csv'

        # define the comined path (if exists) (the folders to save the document)
        self.combined_path = r'G:\study\thesis_data_storage\unordered\classification\set_classification\80k'
        # define the combined path (the actual file)
        self.combined_path_file = r'G:\study\thesis_data_storage\unordered\classification\set_classification\80k\set_classification.csv'

        # define the gridsearch space
        self.param_grid = {'hidden_layer_sizes':[(512, 512, 512, 512), (512, 512, 512)]}

    
    def pre_processor(self, csv_data, target='multiclass'):
        '''
        Input: a csv file from lifetime simulator
        target: a string input that has three options:
            multiclass: four sets
            bandgap_1: whether Et1 large or smaller than midband
            bandgap_2: whether Et2 large or smaller than midband

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

        if target=='multiclass':
            # define y for ML output
            # set 11: 1
            # set 10: 2
            # set 01: 3
            # set 00: 4
            csv_data.set11 = csv_data.apply(lambda row: 1 if row['bandgap_1'] == 1 and row['bandgap_2'] == 1 else 0, axis=1)
            csv_data.set10 = csv_data.apply(lambda row: 2 if row['bandgap_1'] == 1 and row['bandgap_2'] == 0 else 0, axis=1)
            csv_data.set01 = csv_data.apply(lambda row: 3 if row['bandgap_1'] == 0 and row['bandgap_2'] == 1 else 0, axis=1)
            csv_data.set00 = csv_data.apply(lambda row: 4 if row['bandgap_1'] == 0 and row['bandgap_2'] == 0 else 0, axis=1)
            y = csv_data.set11 + csv_data.set10 + csv_data.set01 + csv_data.set00

        elif target == 'bandgap_1':
            y = csv_data['bandgap_1']

        elif target == 'bandgap_2':
            y = csv_data['bandgap_2']

        return X, y


    def train_test_model(self, export_model = True):
        '''
        This function aims to train and test the model
        '''
        # start the timer
        start_time = time.time()

        # load the training data
        training_data = self.combined_data

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

        # report the output
        print(classification_report(y_test, y_pred))

        # confusion matrix
        print(confusion_matrix(y_test, y_pred))

        # store hte trained model into the object
        self.trained_model = model

        # export the model and scaler if required
        if export_model == True:
            path = r'G:\study\thesis_data_storage\journal\set_classification'
            joblib.dump(model, filename = path + '\set_classification.joblib')
            joblib.dump(scaler, filename = path + '\set_classficiation_scaler.joblib')

        # end time
        end_time = time.time()
        dt = end_time - start_time
        print('takes ' + str(dt) + ' seconds')


    def data_combiner(self, export_csv = True):
        '''
        This function take the dataset frmo 4 sests and combine into one data
        '''
        # load the data
        set11_data = pd.read_csv(self.set11_path)
        set10_data = pd.read_csv(self.set10_path)
        set01_data = pd.read_csv(self.set01_path)
        set00_data = pd.read_csv(self.set00_path)

        # sanity check
        # check set 11
        print('Checking set 11')
        print('The number of defects is ' + str(set11_data.shape[0]))
        print('The maximum bandgap 1 value for set 11 is ' + str(set11_data['bandgap_1'].max()))
        print('Whether the column is unique: ' + str(set11_data['bandgap_1'].nunique()))
        print('The maximum bandgap 2 value for set 11 is ' + str(set11_data['bandgap_2'].max()))
        print('Whether the column is unique: ' + str(set11_data['bandgap_2'].nunique()))
        # check set 10
        print('Checking set 10')
        print('The number of defects is ' + str(set10_data.shape[0]))
        print('The maximum bandgap 1 value for set 10 is ' + str(set10_data['bandgap_1'].max()))
        print('Whether the column is unique: ' + str(set10_data['bandgap_1'].nunique()))
        print('The maximum bandgap 2 value for set 10 is ' + str(set10_data['bandgap_2'].max()))
        print('Whether the column is unique: ' + str(set10_data['bandgap_2'].nunique()))
        # check set 01
        print('Checking set 01')
        print('The number of defects is ' + str(set01_data.shape[0]))
        print('The maximum bandgap 1 value for set 01 is ' + str(set01_data['bandgap_1'].max()))
        print('Whether the column is unique: ' + str(set01_data['bandgap_1'].nunique()))
        print('The maximum bandgap 2 value for set 01 is ' + str(set01_data['bandgap_2'].max()))
        print('Whether the column is unique: ' + str(set01_data['bandgap_2'].nunique()))
        # check set 00
        print('Checking set 00')
        print('The number of defects is ' + str(set00_data.shape[0]))
        print('The maximum bandgap 1 value for set 00 is ' + str(set00_data['bandgap_1'].max()))
        print('Whether the column is unique: ' + str(set00_data['bandgap_1'].nunique()))
        print('The maximum bandgap 2 value for set 00 is ' + str(set00_data['bandgap_2'].max()))
        print('Whether the column is unique: ' + str(set00_data['bandgap_2'].nunique()))

        # combine data
        combined_data = pd.concat([set11_data, set10_data, set01_data, set00_data])

        # shuffle
        combined_data = combined_data.sample(frac=1).reset_index(drop=True)

        # store the shuflled data in object
        self.combined_data = combined_data

        # export the combined data
        if export_csv == True:
            path = self.combined_path
            combined_data.to_csv(path + '\set_classification.csv')


    def load_combined_data(self):
        '''
        This function load the pd dataframe from self.combined path
        '''
        self.combined_data = pd.read_csv(self.combined_path_file)


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


    def train_Gridsearch(self):
        '''
        Input: training path
        Output : best parameter and the best score

        '''

        # create the grid search object
        grid_search = GridSearchCV(self.model, param_grid = self.param_grid, cv=5, verbose=3)

        # start the timer
        start_time = time.time()

        # load the training data
        training_data = self.combined_data

        # extract x and y
        X, y = self.pre_processor(training_data)

        # train test split
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        # apply the scaler
        scaler = self.scaler
        scaler.fit(X)
        X_scaled = scaler.transform(X)
    
        # train the grid search object
        print('training gridsearch')
        grid_search.fit(X_scaled, y)

        # print hte best parameters and the best score
        print('The best parameters: ' + str(grid_search.best_params_))
        print('Best score: ', str(grid_search.best_score_))

        # store the best parameters into the object
        self.best_params = grid_search.best_params_

        # end time
        end_time = time.time()
        dt = end_time - start_time
        print('takes ' + str(dt) + ' seconds')