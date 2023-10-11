# %% imports
from residual_map_object import *

# %% Varying Et1

# iterate for Et1_list ranging from -0.5 to 0.5 with a step of 0.1
# plot the graph for this set of parameters
Et1_list = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
Et2_list = Et1_list
se1 = 1e-13
se2 = 1e-13
sp1 = 1e-13
sp2 = 1e-13

# define the object
residual_map = residual_map()

# update the true solution
residual_map.Et1 = -0.5
residual_map.Et2 = -0.5
residual_map.se1 = se1
residual_map.se2 = se2
residual_map.sp1 = sp1
residual_map.sp2 = sp2

# update the data parameters
# residual_map.Doplist = [1e15, 1e15, 1e15]
# residual_map.Tlist = [200, 300, 400]
# residual_map.Typelist=['p', 'p', 'p']

# plot the residual map
residual_map.residual_calculator()

# %% plot the fitting
residual_map.fit_plotter()