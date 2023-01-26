# %% imports
from ML_regression_object import *

# %% use trained model

# define the object
regressor = ML_regression()

# load and test model
regressor.load_test_model()

# %% train a model from scrach
# define the object
regressor = ML_regression()

# train test the model
regressor.train_test_model()

# export the model and scaler
regressor.export_model_scaler()