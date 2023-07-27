# %% imports
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.rcParams['text.usetex'] = True
import numpy as np
# from IPython.display import display, Math
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.font_manager as font_manager
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator
'''
Image standard:
fontsize = 22;
Title font size = 22
figure size = 7 by 7;
'''
# define the parameters
figuresize = (7, 8)
subplots_figuresize = (12, 12)
xtick_size = 22
ytick_size = xtick_size
ylabel_size = 22
title_size = 25
axis_numbersize = 22
xlabel_size = 22
ylabel_size = xlabel_size
text_size=22
chart_number_size = 25
chart_number_position = (0.05, 0.9)
fonttype = "Arial"
legend_size=15
# %% The Figure 2 in Journal
# define the path
path1 = r"D:\study\thesis_data_storage\journal\Figure_plotting_data\Fig_2_mis_defect_classification\Et1_05_Et2_051_two_level.csv"
path2 = r"D:\study\thesis_data_storage\journal\Figure_plotting_data\Fig_2_mis_defect_classification\Et1_05_one_level.csv"
# read the data
one_level = pd.read_csv(path2)
two_level = pd.read_csv(path1)
# plot the data
plt.figure()
plt.plot(two_level.iloc[:, 0], two_level.iloc[:, 1], label='Two level defect', color='blue', alpha=0.5)
plt.plot(one_level.iloc[:, 0], one_level.iloc[:, 1], label='One level defect', linestyle='dashed', color='red', alpha=1)
plt.legend(fontsize=11, loc='lower left', ncol=1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Excess carrer concentration ($\rm cm^{-3}$)', fontsize=xlabel_size)
plt.ylabel('Lifetime (s)', fontsize=ylabel_size)
plt.xticks(fontsize=xtick_size)
plt.yticks([1e-4, 1e-5], fontsize=ytick_size)
# plt.ylim((1e-5, 1e3))
plt.show()
# plot the parts with high difference between lifetime curves
print(np.min(np.array(two_level.iloc[25, 1]) - np.array(one_level.iloc[25, 1]))*1e9)
plt.figure()
plt.plot(two_level.iloc[48:50, 0]*1e-17, two_level.iloc[48:50, 1]*1e5, label='Two level defect', color='blue', alpha=0.5)
plt.plot(one_level.iloc[48:50, 0]*1e-17, one_level.iloc[48:50:, 1]*1e5, label='One level defect', linestyle='dashed', color='red', alpha=1)
# plt.legend(fontsize=11, loc='lower left', ncol=1)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'Excess carrer concentration ($\rm cm^{-3}$)', fontsize=xlabel_size)
# plt.ylabel('Lifetime (s)', fontsize=ylabel_size)
plt.xticks([0.8, 0.9, 1], fontsize=xtick_size)
plt.yticks([1.5, 1.7], fontsize=ytick_size)
# plt.ylim((1e-5, 1e3))
plt.show()
# %% The Figure 3 in Journal
# define the path
path1 = r"D:\study\thesis_data_storage\journal\defect_classification\lifetime_curve_plotting\two_levels.csv"
path2 = r"D:\study\thesis_data_storage\journal\defect_classification\lifetime_curve_plotting\two_one_levels.csv"
# read the data
one_level = pd.read_csv(path2)
two_level = pd.read_csv(path1)
# plot the data
plt.figure()
# plt.plot(two_level.iloc[:, 0], two_level.iloc[:, 1], label='Two level defect', color='blue', alpha=0.5)
plt.plot(one_level.iloc[:, 0], one_level.iloc[:, 1], label='One level defect lifetime ($E_{t}$=$E_{t}$)', color='blue', alpha=1)
# plt.plot(one_level.iloc[:, 0], one_level.iloc[:, 2], label='$E_{t2}$ recombination', linestyle='dashed', color='green', alpha=0.5)
plt.plot(one_level.iloc[:, 0], one_level.iloc[:, 3], label='Effective lifetime of two independent defects', linestyle='dashed', color='red', alpha=1)
plt.legend(fontsize=11, loc='lower left', ncol=1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Excess carrer concentration ($\rm cm^{-3}$)', fontsize=xlabel_size)
plt.ylabel('Lifetime (s)', fontsize=ylabel_size)
plt.xticks(fontsize=xtick_size)
plt.yticks(fontsize=ytick_size)
# plt.ylim((1e-5, 1e3))
plt.show()
# plot the parts with high difference between lifetime curves
print(np.max(np.array(one_level.iloc[:, 1]) - np.array(one_level.iloc[:, 3]))*1e9)
plt.figure()
plt.plot(one_level.iloc[:, 0], np.array(one_level.iloc[:, 1]) - np.array(one_level.iloc[:, 3]))
plt.yscale('log')
plt.show()
# plot the parts with high difference
plt.figure()
plt.plot((one_level.iloc[:2, 0])*1e-12, (one_level.iloc[:2, 1]-1e-3)*1e6, label='$E_{t1}$ one level defect lifetime', color='blue', alpha=1)
plt.plot((one_level.iloc[:2, 0])*1e-12, (one_level.iloc[:2, 3]-1e-3)*1e6, label='Effective lifetime of two independent defects', linestyle='dashed', color='red', alpha=1)
plt.xticks([1.0, 1.1, 1.2], fontsize=xtick_size)
plt.yticks([3.4, 3.5, 3.6], fontsize=ytick_size)
plt.show()
# %% Figure 4 (a) in Journal
# define the path
path_04_01 = r"D:\study\thesis_data_storage\journal\Figure_plotting_data\Fig_4_mis_set_classification\Et_04_01.csv"
path_04_n01 = r"D:\study\thesis_data_storage\journal\Figure_plotting_data\Fig_4_mis_set_classification\Et_04_n01.csv"
path_04_005 = r"D:\study\thesis_data_storage\journal\Figure_plotting_data\Fig_4_mis_set_classification\Et_04_005.csv"
path_04_0 = r"D:\study\thesis_data_storage\journal\Figure_plotting_data\Fig_4_mis_set_classification\Et_04_0.csv"
path_04_n005 = r"D:\study\thesis_data_storage\journal\Figure_plotting_data\Fig_4_mis_set_classification\Et_04_n005.csv"
path_n04_01 = r"D:\study\thesis_data_storage\journal\set_classification\plot_data\Et_n04_01.csv"
path_n04_n01 = r"D:\study\thesis_data_storage\journal\set_classification\plot_data\Et_n04_n01.csv"
path_n04_005 = r"D:\study\thesis_data_storage\journal\Figure_plotting_data\Fig_4_mis_set_classification\Et_n05_005.csv"
path_n04_0 = r"D:\study\thesis_data_storage\journal\Figure_plotting_data\Fig_4_mis_set_classification\Et_n04_0.csv"
path_n04_n005 = r"D:\study\thesis_data_storage\journal\Figure_plotting_data\Fig_4_mis_set_classification\Et_n04_n005.csv"
# read the data
data_04_01 = pd.read_csv(path_04_01)
data_04_n01 = pd.read_csv(path_04_n01)
data_04_005 = pd.read_csv(path_04_005)
data_04_0 = pd.read_csv(path_04_0)
data_04_n005 = pd.read_csv(path_04_n005)
data_n04_01 = pd.read_csv(path_n04_01)
data_n04_n01 = pd.read_csv(path_n04_n01)
data_n04_005 = pd.read_csv(path_n04_005)
data_n04_0 = pd.read_csv(path_n04_0)
data_n04_n005 = pd.read_csv(path_n04_n005)

# Plot the data
fig, axis = plt.subplots(1, 2, figsize=(15, 5))

cmap = cm.get_cmap('viridis')  # Get the YlGn color map
colors = cmap(np.linspace(0, 1, 5))  # Generate five colors from the color map
et2_values = [0.1, 0.05, 0, -0.05, -0.1]  # Et2 values for color scaling

# First subplot
axis[0].plot(data_04_01.iloc[:, 0], data_04_01.iloc[:, 1]*1e6, label='$E_{t1}$=0.4 eV; $E_{t2}$=0.1 eV', color=colors[0])
axis[0].plot(data_04_005.iloc[:, 0], data_04_005.iloc[:, 1]*1e6, label='$E_{t1}$=0.4 eV; $E_{t2}$=0.05 eV', color=colors[1])
axis[0].plot(data_04_0.iloc[:, 0], data_04_0.iloc[:, 1]*1e6, label='$E_{t1}$=0.4 eV; $E_{t2}$=0 eV', color=colors[2])
axis[0].plot(data_04_n005.iloc[:, 0], data_04_n005.iloc[:, 1]*1e6, label='$E_{t1}$=0.4 eV; $E_{t2}$=-0.05 eV', color=colors[3])
axis[0].plot(data_04_n01.iloc[:, 0], data_04_n01.iloc[:, 1]*1e6, label='$E_{t1}$=0.4 eV; $E_{t2}$=-0.1 eV', color=colors[4])
# axis[0].legend(fontsize=legend_size, loc='lower left', ncol=1)
axis[0].set_xscale('log')
axis[0].set_xlabel(r'Excess carrier concentration ($\rm cm^{-3}$)', fontsize=xlabel_size)
axis[0].set_ylabel('Lifetime (µs)', fontsize=ylabel_size)
axis[0].set_xticks([1e12, 1e13, 1e14, 1e15, 1e16, 1e17], fontsize=xtick_size)
axis[0].set_yticks([1, 2, 3, 4], fontsize=ytick_size)
axis[0].text(0.95, 0.95, '(a)', transform=axis[0].transAxes, fontsize=20, va='top', ha='right')
axis[0].tick_params(axis='both', which='major', labelsize=xtick_size)
# Add a color bar for the first subplot
sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(et2_values), max(et2_values)))
sm.set_array([])
cbar = plt.colorbar(sm, ax=axis[0])
cbar.ax.yaxis.set_major_locator(MultipleLocator(0.2))  # Set the major locator for ticks
cbar.set_ticks([-0.1, 0, 0.1])  # Set the tick positions
cbar.set_ticklabels([-0.1, 0, 0.1])  # Set the tick labels
cbar.set_label('$E_{t2}$ values (eV)', fontsize=ytick_size)
cbar.ax.tick_params(labelsize=ytick_size)

# Second subplot
# Second subplot
axis[1].plot(data_n04_01.iloc[:, 0], data_n04_01.iloc[:, 1]*1e6, label='$E_{t1}$=-0.4 eV; $E_{t2}$=0.1 eV', color=colors[0])
axis[1].plot(data_n04_005.iloc[:, 0], data_n04_005.iloc[:, 1]*1e6, label='$E_{t1}$=-0.4 eV; $E_{t2}$=0.05 eV', color=colors[1])
axis[1].plot(data_n04_0.iloc[:, 0], data_n04_0.iloc[:, 1]*1e6, label='$E_{t1}$=-0.4 eV; $E_{t2}$=0 eV', color=colors[2])
axis[1].plot(data_n04_n005.iloc[:, 0], data_n04_n005.iloc[:, 1]*1e6, label='$E_{t1}$=-0.4 eV; $E_{t2}$=-0.05 eV', color=colors[3])
axis[1].plot(data_n04_n01.iloc[:, 0], data_n04_n01.iloc[:, 1]*1e6, label='$E_{t1}$=-0.4 eV; $E_{t2}$=-0.1 eV', color=colors[4])
# axis[1].legend(fontsize=legend_size, loc='lower left', ncol=1)
axis[1].set_xscale('log')
axis[1].set_xlabel(r'Excess carrier concentration ($\rm cm^{-3}$)', fontsize=xlabel_size)
axis[1].set_ylabel('Lifetime (µs)', fontsize=ylabel_size)
axis[1].set_xticks([1e12, 1e13, 1e14, 1e15, 1e16, 1e17], fontsize=xtick_size)
axis[1].set_yticks([0.1, 0.5, 1], fontsize=ytick_size)
axis[1].text(0.95, 0.95, '(b)', transform=axis[1].transAxes, fontsize=20, va='top', ha='right')
axis[1].tick_params(axis='both', which='major', labelsize=xtick_size)

# Add a color bar for the second subplot
sm2 = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(et2_values), max(et2_values)))
sm2.set_array([])
cbar2 = plt.colorbar(sm2, ax=axis[1])
cbar2.ax.yaxis.set_major_locator(MultipleLocator(0.2))  # Set the major locator for ticks
cbar2.set_ticks([-0.1, 0, 0.1])  # Set the tick positions
cbar2.set_ticklabels([-0.1, 0, 0.1])  # Set the tick labels
cbar2.set_label('$E_{t2}$ values (eV)', fontsize=ytick_size)
cbar2.ax.tick_params(labelsize=ytick_size)

# adjust spacing between subplots
plt.subplots_adjust(wspace=0.5)

plt.show()


# # plot the difference between the lifetime to pick where to zoom in
# plt.figure()
# # Calculate the difference in lifetime values
# diff_lifetime = data_04_01.iloc[:, 1]*1e6 - data_04_n01.iloc[:, 1]*1e6
# diff_lifetime2 = data_n04_01.iloc[:, 1]*1e6 - data_n04_n01.iloc[:, 1]*1e6
# # Create a new figure and axis for the difference plot
# fig_diff, axis_diff = plt.subplots(figsize=(8, 5))
# # Plot the difference in lifetime
# axis_diff.plot(data_04_01.iloc[:, 0], diff_lifetime, label='Lifetime Difference in first plot', color='red')
# axis_diff.plot(data_04_01.iloc[:, 0], diff_lifetime2, label='Lifetime Difference in second plot', color='blue')
# # Set the x-axis and y-axis labels for the difference plot
# axis_diff.set_xlabel(r'Excess carrier concentration ($\rm cm^{-3}$)', fontsize=xlabel_size)
# axis_diff.set_ylabel('Lifetime Difference (µs)', fontsize=ylabel_size)
# # Add a title for the difference plot
# axis_diff.set_title('Lifetime Difference', fontsize=title_size)
# # Add a legend for the difference plot
# axis_diff.legend(fontsize=legend_size, loc='upper right')
# # Show the plot
# plt.show()


# # Convert the columns of interest to NumPy arrays

# data_04_01_values = data_04_01.iloc[:, 1].values
# data_04_n01_values = data_04_n01.iloc[:, 1].values

# # Calculate the absolute difference between the two arrays
# diff = np.abs(data_04_01_values - data_04_n01_values)

# # Find the indices with the maximum difference
# max_indices = np.argsort(diff)[-2:]  # Get the indices of the two maximum differences
# print(max_indices)
# # Get the corresponding x-values using iloc
# x_values = data_04_01.iloc[:, 0].values

# # Get the x-values at the two maximum difference indices
# x_max1 = x_values[max_indices[0]]
# x_max2 = x_values[max_indices[1]]
# print(np.log10(x_max1))
# print(np.log10(x_max2))


# Plot the zoom in data
fig, axis = plt.subplots(1, 2, figsize=(15, 5))

cmap = cm.get_cmap('viridis')  # Get the YlGn color map
colors = cmap(np.linspace(0, 1, 5))  # Generate five colors from the color map
et2_values = [0.1, 0.05, 0, -0.05, -0.1]  # Et2 values for color scaling

# First subplot
axis[0].plot(data_04_01.iloc[33:35, 0], data_04_01.iloc[33:35, 1]*1e6, label='$E_{t1}$=0.4 eV; $E_{t2}$=0.1 eV', color=colors[0])
axis[0].plot(data_04_005.iloc[33:35, 0], data_04_005.iloc[33:35, 1]*1e6, label='$E_{t1}$=0.4 eV; $E_{t2}$=0.05 eV', color=colors[1])
axis[0].plot(data_04_0.iloc[33:35, 0], data_04_0.iloc[33:35, 1]*1e6, label='$E_{t1}$=0.4 eV; $E_{t2}$=0 eV', color=colors[2])
axis[0].plot(data_04_n005.iloc[33:35, 0], data_04_n005.iloc[33:35, 1]*1e6, label='$E_{t1}$=0.4 eV; $E_{t2}$=-0.05 eV', color=colors[3])
axis[0].plot(data_04_n01.iloc[33:35, 0], data_04_n01.iloc[33:35, 1]*1e6, label='$E_{t1}$=0.4 eV; $E_{t2}$=-0.1 eV', color=colors[4])
# axis[0].legend(fontsize=legend_size, loc='lower left', ncol=1)
# axis[0].set_xscale('log')
axis[0].set_xlabel(r'Excess carrier concentration ($\rm cm^{-3}$)', fontsize=xlabel_size)
axis[0].set_ylabel('Lifetime (µs)', fontsize=ylabel_size)
# axis[0].set_xticks([1e12, 1e13, 1e14, 1e15, 1e16, 1e17], fontsize=xtick_size)
# axis[0].set_yticks([1, 2, 3, 4], fontsize=ytick_size)
# axis[0].text(0.95, 0.95, '(a)', transform=axis[0].transAxes, fontsize=20, va='top', ha='right')
# axis[0].tick_params(axis='both', which='major', labelsize=xtick_size)
# Add a color bar for the first subplot
sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(et2_values), max(et2_values)))
sm.set_array([])
cbar = plt.colorbar(sm, ax=axis[0])
cbar.ax.yaxis.set_major_locator(MultipleLocator(0.2))  # Set the major locator for ticks
cbar.set_ticks([-0.1, 0, 0.1])  # Set the tick positions
cbar.set_ticklabels([-0.1, 0, 0.1])  # Set the tick labels
cbar.set_label('$E_{t2}$ values (eV)', fontsize=ytick_size)
cbar.ax.tick_params(labelsize=ytick_size)

# Second subplot
axis[1].plot(data_n04_01.iloc[:, 0], data_n04_01.iloc[:, 1]*1e6, label='$E_{t1}$=-0.4 eV; $E_{t2}$=0.1 eV', color=colors[0])
axis[1].plot(data_n04_005.iloc[:, 0], data_n04_005.iloc[:, 1]*1e6, label='$E_{t1}$=-0.4 eV; $E_{t2}$=0.05 eV', color=colors[1])
axis[1].plot(data_n04_0.iloc[:, 0], data_n04_0.iloc[:, 1]*1e6, label='$E_{t1}$=-0.4 eV; $E_{t2}$=0 eV', color=colors[2])
axis[1].plot(data_n04_n005.iloc[:, 0], data_n04_n005.iloc[:, 1]*1e6, label='$E_{t1}$=-0.4 eV; $E_{t2}$=-0.05 eV', color=colors[3])
axis[1].plot(data_n04_n01.iloc[:, 0], data_n04_n01.iloc[:, 1]*1e6, label='$E_{t1}$=-0.4 eV; $E_{t2}$=-0.1 eV', color=colors[4])
# axis[1].legend(fontsize=legend_size, loc='lower left', ncol=1)
axis[1].set_xscale('log')
axis[1].set_xlabel(r'Excess carrier concentration ($\rm cm^{-3}$)', fontsize=xlabel_size)
axis[1].set_ylabel('Lifetime (µs)', fontsize=ylabel_size)
axis[1].set_xticks([1e12, 1e13, 1e14, 1e15, 1e16, 1e17], fontsize=xtick_size)
axis[1].set_yticks([0.1, 0.5, 1], fontsize=ytick_size)
axis[1].text(0.95, 0.95, '(b)', transform=axis[1].transAxes, fontsize=20, va='top', ha='right')
axis[1].tick_params(axis='both', which='major', labelsize=xtick_size)

# Add a color bar for the second subplot
sm2 = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(et2_values), max(et2_values)))
sm2.set_array([])
cbar2 = plt.colorbar(sm2, ax=axis[1])
cbar2.ax.yaxis.set_major_locator(MultipleLocator(0.2))  # Set the major locator for ticks
cbar2.set_ticks([-0.1, 0, 0.1])  # Set the tick positions
cbar2.set_ticklabels([-0.1, 0, 0.1])  # Set the tick labels
cbar2.set_label('$E_{t2}$ values (eV)', fontsize=ytick_size)
cbar2.ax.tick_params(labelsize=ytick_size)

# adjust spacing between subplots
plt.subplots_adjust(wspace=0.5)
plt.show()

# to do: change to 500000 data point then add width