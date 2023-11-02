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
regressor.training_path = r"D:\study\thesis_data_storage\journal\defect_classification\basic_model\testing_data\outputs\2023-10-14-12-31-30_advanced example - multi_level_L_datasetID_0.csv"
regressor.y_str = 'Et_eV_1'

# train test the model
regressor.train_test_model(apply_PCA=False)

# export the model and scaler
regressor.export_model_scaler()

# %% train a model for multioutput
# define the object
regressor = ML_regression()

# define the training path
regressor.training_path = r"D:\study\thesis_data_storage\journal\defect_classification\basic_model\testing_data\outputs\Set01_8k.csv"
# define the ML model
regressor.model = MLPRegressor(hidden_layer_sizes=(50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                                                   50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                                                   50, 50, 50, 50, 50, 50, 50, 50, 50, 50), tol=1e-7, random_state=1, verbose=True, batch_size=32)
# regressor.model = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=1, verbose=True)

# define thee grid serach parameters for NN
regressor.param_dist = {
    'hidden_layer_sizes': [(50 ,50, 50)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.0001],
    'learning_rate': ['constant'],
    'learning_rate_init': [0.001],
}

# define the y variable
regressor.y_str = 'Multi-output'

# define the coluor code
regressor.colour_str = 'logSp_2'

# train test the model
regressor.train_test_model_multi(apply_PCA=False, randomsearch=False)

# export the model and scaler
regressor.export_model_scaler()