# %% imports
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
for central_Et2 in np.arange(-0.5, 0.6, 0.1):
    print('up to ' + str(central_Et2))
    # define the object
    simulator = two_level_lifetime_generator()
    # define the simulation range

    simulator.PARAM['Et_min_2'] = central_Et2 - 0.05
    simulator.PARAM['Et_max_2'] = central_Et2 + 0.05
    # define the file name
    simulator.NAME = 'multi_level_L' + 'Et1' + str(central_Et2)
    simulator.mixture_simulate()

# %% simulate the defect classification dataset
simulator = two_level_lifetime_generator()
simulator.PARAMETERS['n_defects']=8000
# update the save directory
simulator.SAVEDIR = r'G:\study\thesis_data_storage\unordered\set00\p\set00_8k'
# update Et range
simulator.PARAM['Et_min_1'] = 0
simulator.PARAM['Et_max_1'] = 0.55
simulator.PARAM['Et_min_2'] = -0.55
simulator.PARAM['Et_max_2'] = 0
# simulate the data
simulator.mixture_simulate()

# %% Simulate two-level-lifetime only
simulator = two_level_lifetime_generator()
simulator.PARAMETERS['n_defects']=80000
# update the save directory
simulator.SAVEDIR = r'G:\study\thesis_data_storage\unordered\set00\p\set00_80k'
# update Et range
simulator.PARAM['Et_min_1'] = -0.55
simulator.PARAM['Et_max_1'] = 0
simulator.PARAM['Et_min_2'] = -0.55
simulator.PARAM['Et_max_2'] = 0
# simulate the data
simulator.two_level_simulate()

# %%
simulator.email_reminder()