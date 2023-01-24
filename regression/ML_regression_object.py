# %% imports
import numpy as np

# %% define the object

class ML_regression():
    '''
    This object aim to use ML regression to predict the 6 parameter for two level defect based on Sah-Shockley Equation
    '''

    def __init__(self):
        # define the path for the training dataset
        # self.training_path = 