# %% imports
from ML_regression_object import *

# %% use trained model

# define the object
regressor = ML_regression()

# define the y for regression
regressor.y_str = 'Et_eV_2'

# define the testing path
regressor.testing_path = r"D:\study\thesis_data_storage\journal\defect_classification\basic_model\testing_data\outputs\2023-10-06-08-53-23_advanced example - multi_level_L_datasetID_0.csv"

# define the model pathS
regressor.model_path = r'C:\Users\sijin wang\Documents\GitHub\Two_level_defect_journal\regression\trained_model_Et_eV_2.joblib'

# define the scater path
regressor.scaler_path = r'C:\Users\sijin wang\Documents\GitHub\Two_level_defect_journal\regression\trained_scaler_Et_eV_2.joblib'

# load and test model
regressor.load_test_model()

# %% train a model from given directory
# define the object
regressor = ML_regression()

# define the training path
regressor.training_path = r"D:\study\thesis_data_storage\journal\defect_classification\basic_model\testing_data\outputs\2023-10-06-08-29-25_advanced example - multi_level_L_datasetID_0.csv"
# define the y variable
regressor.y_str = 'Et_eV_2'

# train test the model
regressor.train_test_model()

# export the model and scaler
regressor.export_model_scaler()