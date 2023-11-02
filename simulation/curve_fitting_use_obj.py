# %% imports
import sys
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\Two_level_defect_journal\simulation')
from curve_fitting_obj import *

# %% fit the curve: swing all variables
fitter = experiment_fitter()
# interpolate the data
fitter.experimental_data_interpolator(plot=True ,polyorder_n=1, wind_l_n=91, polyorder_p=2, wind_l_p=81)

# update the save directory
fitter.PARAMETERS['n_defects'] = 1
# simualte the data
fitter.experimental_data_fit2()

# %% fit the curve: keep Yan solution
fitter = experiment_fitter()
# interpolate the data
fitter.experimental_data_interpolator(plot=True ,polyorder_n=1, wind_l_n=91, polyorder_p=2, wind_l_p=81)
# update the save directory
fitter.PARAMETERS['n_defects'] = 1
# simualte the data
fitter.experimental_data_fit3()