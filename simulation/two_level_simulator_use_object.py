# %% imports
import sys
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\Two_level_defect_journal\simulation')
from two_level_simulator_object import *

# define the object
# simulator = two_level_lifetime_generator()

# %%simulate the data with a range of Et1

# swtip Et2 from -0.5 to 0.5 eV
# for central_Et1 in np.arange(-0.5, 0.6, 0.1):
#     print('up to ' + str(central_Et1))
#     # define the object
#     simulator = two_level_lifetime_generator()
#     # define the simulation range

#     simulator.PARAM['Et_min_1'] = central_Et1 - 0.05
#     simulator.PARAM['Et_max_1'] = central_Et1 + 0.05
#     # define the file name
#     simulator.NAME = 'multi_level_L' + 'Et1' + str(central_Et1)
#     simulator.mixture_simulate()

# swtip Et2 from -0.5 to 0.5 eV
# for central_Et2 in np.arange(-0.5, 0.6, 0.1):
#     print('up to ' + str(central_Et2))
#     # define the object
#     simulator = two_level_lifetime_generator()
#     # define the simulation range

#     simulator.PARAM['Et_min_2'] = central_Et2 - 0.05
#     simulator.PARAM['Et_max_2'] = central_Et2 + 0.05
#     # define the file name
#     simulator.NAME = 'multi_level_L' + 'Et1' + str(central_Et2)
#     simulator.mixture_simulate()

# %% simulate the defect classification dataset
simulator = two_level_lifetime_generator()
simulator.PARAMETERS['n_defects']=800
# update the save directory
simulator.SAVEDIR = r"D:\study\thesis_data_storage\journal\defect_classification\basic_model\testing_data"
# update Et range
simulator.PARAM['Et_min_1'] = -0.55
simulator.PARAM['Et_max_1'] = 0.55
simulator.PARAM['Et_min_2'] = -0.55
simulator.PARAM['Et_max_2'] = 0.55
# simulate the data
simulator.mixture_simulate()

# %% Simulate two-level-lifetime only
simulator = two_level_lifetime_generator()
# define the number of defect to simulate
simulator.PARAMETERS['n_defects']= 1
# update the save directory
simulator.SAVEDIR = r"D:\study\thesis_data_storage\journal\defect_classification\basic_model\testing_data"
# update Et range
simulator.PARAM['Et_min_1'] = -0.55
simulator.PARAM['Et_max_1'] = 0.55
simulator.PARAM['Et_min_2'] = -0.55
simulator.PARAM['Et_max_2'] = 0.55
# simulate the data
simulator.two_level_simulate()

# %% Simulate the Yan BO lifetime
simulator = two_level_lifetime_generator()
# define the number of defect to simulate
simulator.PARAMETERS['n_defects'] = 1
# update the save directory
simulator.SAVEDIR = r"D:\study\thesis_data_storage\journal\defect_classification\basic_model\testing_data"
# simualte the data
simulator.two_level_simulate_BO()

# %% Simulate training dataset n 303K and p 303 K
simulator = two_level_lifetime_generator()
# define the number of defect to simulate
simulator.PARAMETERS['n_defects'] = 1
# update the save directory
simulator.SAVEDIR = r"D:\study\thesis_data_storage\journal\defect_classification\basic_model\testing_data"
# simualte the data
simulator.BO_training_simulator_2_curves(emailme=False)

# %% Interpolate the expeirmental data
simulator = two_level_lifetime_generator()
# interpolate the data
simulator.experimental_data_interpolator(plot=True)

# export the data
# update the save directory
simulator.PARAMETERS['n_defects'] = 1
simulator.SAVEDIR = r"D:\study\thesis_data_storage\journal\defect_classification\basic_model\testing_data"
# simualte the data
simulator.interpolate_data_exporter()

# %% find the sp1 and sp2
# Simulate the Yan BO lifetime
simulator = two_level_lifetime_generator()
# define the number of defect to simulate
simulator.PARAMETERS['n_defects'] = 1
# update the save directory
simulator.SAVEDIR = r"D:\study\thesis_data_storage\journal\defect_classification\basic_model\testing_data"
# simualte the data
simulator.experimental_data_fit(sp1_resolution=5, sp2_resolution=5, Ntn_resolution=6, Ntp_resolution=6)

# %% Simulate the Yan BO lifetime with optimized scan method.
simulator = two_level_lifetime_generator()
# define the number of defect to simulate
simulator.PARAMETERS['n_defects'] = 1
# update the save directory
simulator.SAVEDIR = r"D:\study\thesis_data_storage\journal\defect_classification\basic_model\testing_data"
# simualte the data
simulator.experimental_data_fit2()

# %% create Yan fitting
# %% Interpolate the expeirmental data
simulator = two_level_lifetime_generator()
# interpolate the data
simulator.experimental_data_interpolator(plot=True ,polyorder_n=1, wind_l_n=91, polyorder_p=2, wind_l_p=81)

# export the data
# update the save directory
simulator.PARAMETERS['n_defects'] = 1
simulator.SAVEDIR = r"D:\study\thesis_data_storage\journal\defect_classification\basic_model\testing_data"
# simualte the data
simulator.interpolate_data_exporter()
# %%
simulator = two_level_lifetime_generator()
simulator.email_reminder()