# %% imports
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.rcParams['text.usetex'] = True
import numpy as np
# from IPython.display import display, Math
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.font_manager as font_manager
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
'''
Image standard:
fontsize = 22;
Title font size = 22
figure size = 7 by 7;
'''
# define the parameters
figuresize = (7, 8)
subplots_figuresize = (12, 12)
xtick_size = 18
ytick_size = xtick_size
title_size = 25
axis_numbersize = 18
xlabel_size = 18
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
plt.plot(two_level.iloc[:, 0], two_level.iloc[:, 1], label='Two-level defect', color='blue', alpha=0.5)
plt.plot(one_level.iloc[:, 0], one_level.iloc[:, 1], label='One-level defect', linestyle='dashed', color='red', alpha=1)
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
plt.plot(one_level.iloc[:, 0], one_level.iloc[:, 1], label='One level defect lifetime', color='blue', alpha=1)
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


# %% Figure 4 in Journal
# define the path for My dell laptop
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

# Set font sizes
xlabel_size = 18
ylabel_size = 18
xtick_size = 18
ytick_size = 18

# Define colors for the plots
cmap = cm.get_cmap('coolwarm')
colors = cmap(np.linspace(0, 1, 5))

# Create a figure and axis
fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot the data
ax1.plot(data_04_01.iloc[:, 0], data_04_01.iloc[:, 1]*1e6, label='$E_{t1}$=0.4 eV; $E_{t2}$=0.1 eV', color=colors[0])
ax1.plot(data_04_005.iloc[:, 0], data_04_005.iloc[:, 1]*1e6, label='$E_{t1}$=0.4 eV; $E_{t2}$=0.05 eV', color=colors[1])
ax1.plot(data_04_0.iloc[:, 0], data_04_0.iloc[:, 1]*1e6, label='$E_{t1}$=0.4 eV; $E_{t2}$=0 eV', color=colors[2])
ax1.plot(data_04_n005.iloc[:, 0], data_04_n005.iloc[:, 1]*1e6, label='$E_{t1}$=0.4 eV; $E_{t2}$=-0.05 eV', color=colors[3])
ax1.plot(data_04_n01.iloc[:, 0], data_04_n01.iloc[:, 1]*1e6, label='$E_{t1}$=0.4 eV; $E_{t2}$=-0.1 eV', color=colors[4])

# Set the axes to a logarithmic scale and specify y-axis tick values
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_yticks([10**0, 10**1])

# Format y-axis tick labels as powers of 10
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

# Set labels and ticks
ax1.set_xlabel(r'Excess carrier concentration ($\rm cm^{-3}$)', fontsize=xlabel_size)
ax1.set_ylabel('Lifetime (µs)', fontsize=ylabel_size)
ax1.set_xticks([1e12, 1e13, 1e14, 1e15, 1e16, 1e17])
ax1.tick_params(axis='both', which='major', labelsize=xtick_size)

# Remove grid
ax1.grid(False)

# Create a color map for the color bar
sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-0.1, 0.1))
sm.set_array([])

# Add a color bar
cbar = plt.colorbar(sm, ax=ax1)
cbar.ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
cbar.set_ticks([-0.1, 0, 0.1])
cbar.set_ticklabels([-0.1, 0, 0.1])
cbar.set_label('$E_{t2}$ values (eV)', fontsize=ytick_size)
cbar.ax.tick_params(labelsize=ytick_size)

# Show the plot
plt.show()


# %% Plot the zoom in data for figure 4
plt.figure(figsize=(10, 6))  # Set the figure size

colour_number = 1200
starting_index = 33
ending_index = 35
cmap = cm.get_cmap('coolwarm')  # Get the YlGn color map
colors = cmap(np.linspace(0, 1, colour_number))  # Generate a number of colors from the color map
et2_values = [0.1, 0.05, 0, -0.05, -0.1]  # Et2 values for color scaling

# Plot the original lines
plt.plot(data_04_01.iloc[starting_index:ending_index, 0], data_04_01.iloc[starting_index:ending_index, 1]*1e6, label='$E_{t1}$=0.4 eV; $E_{t2}$=0.1 eV', color=colors[0])
plt.plot(data_04_n01.iloc[starting_index:ending_index, 0], data_04_n01.iloc[starting_index:ending_index, 1]*1e6, label='$E_{t1}$=0.4 eV; $E_{t2}$=-0.1 eV', color=colors[-1])

# Equally fill the space between the two lines with 10 more lines that are parallel
num_lines = colour_number - 2
for i in range(1, num_lines + 1):
    alpha_value = i / (num_lines + 1)  # Determine the alpha value for color and transparency
    y_values = (1 - alpha_value) * data_04_01.iloc[starting_index:ending_index, 1]*1e6 + alpha_value * data_04_n01.iloc[starting_index:ending_index, 1]*1e6
    plt.plot(data_04_01.iloc[starting_index:ending_index, 0], y_values, color=colors[i+1], alpha=alpha_value)

# Axis setting
# plt.xlabel(r'Excess carrier concentration ($\rm cm^{-3}$)', fontsize=xlabel_size)
# plt.ylabel('Lifetime (µs)', fontsize=ylabel_size)

# Set the x-axis to show only 3 ticks
num_ticks = 3
plt.gca().xaxis.set_major_locator(MaxNLocator(num_ticks))
plt.tick_params(axis='both', labelsize=ytick_size*2)

# Add colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(et2_values), max(et2_values)))
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.ax.yaxis.set_major_locator(MultipleLocator(0.2))  # Set the major locator for ticks
cbar.set_ticks([-0.1, 0, 0.1])  # Set the tick positions
cbar.set_ticklabels([-0.1, 0, 0.1])  # Set the tick labels
cbar.set_label('$E_{t2}$ values (eV)', fontsize=ytick_size)
cbar.ax.tick_params(labelsize=ytick_size)

plt.show()  # Show the plot

# %% Figure 6 of the journal article

# define the path for dell Laptop
# path_045_040 = r"D:\study\thesis_data_storage\journal\Figure_plotting_data\Fig_6_regression\Fig_6a\Et_045_040.csv"
# path_045_050 = r"D:\study\thesis_data_storage\journal\Figure_plotting_data\Fig_6_regression\Fig_6a\Et_045_050.csv"

# define the path for office PC
path_045_040 = r"G:\study\thesis_data_storage\journal\Figure_plotting_data\Fig_6_regression\Fig_6a\Et_045_040.csv"
path_045_050 = r"G:\study\thesis_data_storage\journal\Figure_plotting_data\Fig_6_regression\Fig_6a\Et_045_050.csv"

# import the data
data_045_040 = pd.read_csv(path_045_040)
data_045_050 = pd.read_csv(path_045_050)

# Plot the data
fig, axis = plt.subplots(1, 2, figsize=(15, 5))

# define the parameters
colour_number = 100
cmap = cm.get_cmap('viridis')  # Get the YlGn color map
colors = cmap(np.linspace(0, 1, colour_number))  # Generate five colors from the color map
et2_values = [0.45, 0.50, 0.55]  # Et2 values for color scaling

# First subplot
axis[0].plot(data_045_040.iloc[:, 0], data_045_040.iloc[:, 1]*1e6, label='$E_{t1}$=0.4 eV; $E_{t2}$=0.1 eV', color=colors[0])
axis[0].plot(data_045_050.iloc[:, 0], data_045_050.iloc[:, 1]*1e6, label='$E_{t1}$=0.4 eV; $E_{t2}$=0.05 eV', color=colors[-1])

axis[0].set_xscale('log')
axis[0].set_xlabel(r'Excess carrier concentration ($\rm cm^{-3}$)', fontsize=xlabel_size)
axis[0].set_ylabel('Lifetime (µs)', fontsize=ylabel_size)
axis[0].set_xticks([1e12, 1e13, 1e14, 1e15, 1e16, 1e17], fontsize=xtick_size)
axis[0].set_yticks([5, 10, 15, 20], fontsize=ytick_size)
axis[0].text(0.95, 0.95, '(a)', transform=axis[0].transAxes, fontsize=20, va='top', ha='right')
axis[0].tick_params(axis='both', which='major', labelsize=xtick_size)
# Add a color bar for the first subplot
sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(et2_values), max(et2_values)))
sm.set_array([])
cbar = plt.colorbar(sm, ax=axis[0])
cbar.ax.yaxis.set_major_locator(MultipleLocator(0.2))  # Set the major locator for ticks
cbar.set_ticks([0.45, 0.50, 0.55])  # Set the tick positions
cbar.set_ticklabels([0.45, 0.50, 0.55])  # Set the tick labels
cbar.set_label('$E_{t2}$ values (eV)', fontsize=ytick_size)
cbar.ax.tick_params(labelsize=ytick_size)

# Code for equally filling the space between the two lines in Plot A
num_lines = colour_number - 2
for i in range(1, num_lines + 1):
    alpha_value = i / (num_lines + 1)  # Determine the alpha value for color and transparency
    y_values = (1 - alpha_value) * data_045_040.iloc[:, 1] * 1e6 + alpha_value * data_045_050.iloc[:, 1] * 1e6
    axis[0].plot(data_045_040.iloc[:, 0], y_values, color=colors[i+1], alpha=alpha_value)


# Repeat the same process for subplot B
# Define the path
path_050_045 = r"D:\study\thesis_data_storage\journal\Figure_plotting_data\Fig_6_regression\Fig_6b\Et_050_045.csv"
path_050_055 = r"D:\study\thesis_data_storage\journal\Figure_plotting_data\Fig_6_regression\Fig_6b\Et_050_055.csv"

# Import the data
data_050_045 = pd.read_csv(path_050_045)
data_050_055 = pd.read_csv(path_050_055)

# Set the layout to have wider spacing between subplots
plt.subplots_adjust(wspace=0.6)

# Plot B
# define the parameters for plot B
colour_number_plot_b = 10000
cmap_b = cm.get_cmap('viridis')
colors_b = cmap_b(np.linspace(0, 1, colour_number_plot_b))
et2_values_b = [0.45, 0.50, 0.55]

# Second subplot
axis[1].plot(data_050_045.iloc[:, 0], data_050_045.iloc[:, 1]*1e6, label='$E_{t1}$=0.5 eV; $E_{t2}$=0.05 eV', color=colors_b[0])
axis[1].plot(data_050_055.iloc[:, 0], data_050_055.iloc[:, 1]*1e6, label='$E_{t1}$=0.5 eV; $E_{t2}$=0.55 eV', color=colors_b[-1])

axis[1].set_xscale('log')
axis[1].set_xlabel(r'Excess carrier concentration ($\rm cm^{-3}$)', fontsize=xlabel_size)
axis[1].set_ylabel('Lifetime (µs)', fontsize=ylabel_size)
axis[1].set_xticks([1e12, 1e13, 1e14, 1e15, 1e16, 1e17], fontsize=xtick_size)
axis[1].set_yticks([200, 400, 600, 800, 1000], fontsize=ytick_size)
axis[1].text(0.95, 0.95, '(b)', transform=axis[1].transAxes, fontsize=20, va='top', ha='right')
axis[1].tick_params(axis='both', which='major', labelsize=xtick_size)

# Code for equally filling the space between the two lines in Plot B
num_lines_b = colour_number_plot_b - 2
for i in range(1, num_lines_b + 1):
    alpha_value = i / (num_lines_b + 1)
    y_values = (1 - alpha_value) * data_050_045.iloc[:, 1] * 1e6 + alpha_value * data_050_055.iloc[:, 1] * 1e6
    axis[1].plot(data_050_045.iloc[:, 0], y_values, color=colors_b[i], alpha=alpha_value)

# Add a color bar for the second subplot with actual et2_values range
et2_values_actual_b = [0.45, 0.55]
sm2 = cm.ScalarMappable(cmap=cmap_b, norm=plt.Normalize(et2_values_actual_b[0], et2_values_actual_b[1]))
sm2.set_array([])
cbar2 = plt.colorbar(sm2, ax=axis[1])
cbar2.set_ticks([0.45, 0.50, 0.55])
cbar2.set_ticklabels([0.45, 0.50, 0.55])
cbar2.ax.tick_params(labelsize=ytick_size)
cbar2.set_label('$E_{t1}$ values (eV)', fontsize=ytick_size)

plt.show()  # Show the plot

# %% Figure 5: regression true vs prediction
# Set 11 p type.
path_Et1_11_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\Et_eV_1set11_800k.csv'
path_Et2_11_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\Et_eV_2set11_800k.csv'
path_Sn1_11_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\logSn_1set11_800k.csv'
path_Sn2_11_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\logSn_2set11_800k.csv'
path_Sp1_11_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\logSp_1set11_800k.csv'
path_Sp2_11_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\logSp_2set11_800k.csv'
path_k1_11_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\logk_1set11_800k.csv'
path_k2_11_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\logk_2set11_800k.csv'
# Set 11 n type.
path_Et1_11_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\Et_eV_1set11_800k_n.csv'
path_Et2_11_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\Et_eV_2set11_800k_n.csv'
path_Sn1_11_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\logSn_1set11_800k_n.csv'
path_Sn2_11_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\logSn_2set11_800k_n.csv'
path_Sp1_11_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\logSp_1set11_800k_n.csv'
path_Sp2_11_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\logSp_2set11_800k_n.csv'
path_k1_11_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\logk_1set11_800k_n.csv'
path_k2_11_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\logk_2set11_800k_n.csv'
# Set 10 p type.
path_Et1_10_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\Re__ML_finish_training\Et_eV_12022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path_Et2_10_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\Re__ML_finish_training\Et_eV_22022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path_Sn1_10_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\Re__ML_finish_training\logSn_12022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path_Sn2_10_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\Re__ML_finish_training\logSn_22022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path_Sp1_10_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\Re__ML_finish_training\logSp_12022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path_Sp2_10_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\Re__ML_finish_training\logSp_22022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path_k1_10_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\logk_12022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path_k2_10_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\logk_22022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
# Set 10 n type.
path_Et1_10_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\Re__\Et_eV_1set10_800k_n.csv'
path_Et2_10_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\Re__\Et_eV_2set10_800k_n.csv'
path_Sn1_10_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\Re__\logSn_1set10_800k_n.csv'
path_Sn2_10_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\Re__\logSn_2set10_800k_n.csv'
path_Sp1_10_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\Re__\logSp_1set10_800k_n.csv'
path_Sp2_10_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\Re__\logSp_2set10_800k_n.csv'
path_k1_10_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\logk_1set10_800k_n.csv'
path_k2_10_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\logk_2set10_800k_n.csv'
# Set 01 p type.
path_Et1_01_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\Et_eV_1set01_800k_p.csv'
path_Et2_01_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\Et_eV_2set01_800k_p.csv'
path_k1_01_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\logk_1set01_800k_p.csv'
path_k2_01_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\logk_2set01_800k_p.csv'
path_Sn1_01_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\logSn_1set01_800k_p.csv'
path_Sn2_01_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\logSn_2set01_800k_p.csv'
path_Sp1_01_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\logSp_1set01_800k_p.csv'
path_Sp2_01_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\logSp_2set01_800k_p.csv'
# Set 01 n type.
path_Et1_01_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\n\set01_n_800k\Et_eV_1set01_800k_n.csv'
path_Et2_01_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\n\set01_n_800k\Et_eV_2set01_800k_n.csv'
path_Sn1_01_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\n\set01_n_800k\logSn_1set01_800k_n.csv'
path_Sn2_01_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\n\set01_n_800k\logSn_2set01_800k_n.csv'
path_Sp1_01_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\n\set01_n_800k\logSp_1set01_800k_n.csv'
path_Sp2_01_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\n\set01_n_800k\logSp_2set01_800k_n.csv'
path_k1_01_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\n\logk_1set01_800k_n.csv'
path_k2_01_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\n\logk_2set01_800k_n.csv'
# Set 00 p type.
path_Et1_00_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\Et_eV_1set00_800k.csv'
path_Et2_00_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\Et_eV_2set00_800k.csv'
path_k1_00_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logk_1set00_800k.csv'
path_k2_00_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logk_2set00_800k.csv'
path_Sn1_00_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logSn_1set00_800k.csv'
path_Sn2_00_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logSn_2set00_800k.csv'
path_Sp1_00_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logSp_1set00_800k.csv'
path_Sp2_00_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logSp_2set00_800k.csv'
# Set 00 n type.
path_Et1_00_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\Et_eV_1set00_n_800k.csv'
path_Et2_00_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\Et_eV_2set00_n_800k.csv'
path_k1_00_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\logk_1set00_n_800k.csv'
path_k2_00_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\logk_2set00_n_800k.csv'
path_Sn1_00_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\logSn_1set00_n_800k.csv'
path_Sn2_00_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\logSn_2set00_n_800k.csv'
path_Sp1_00_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\logSp_1set00_n_800k.csv'
path_Sp2_00_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\logSp_2set00_n_800k.csv'
# put everything in a list:
# both n and p.
# Et1list = [path_Et1_11_p, path_Et1_11_n, path_Et1_10_p, path_Et1_10_n, path_Et1_01_p, path_Et1_01_n, path_Et1_00_p, path_Et1_00_n]
# Et2list = [path_Et2_11_p, path_Et2_11_n, path_Et2_10_p, path_Et2_10_n, path_Et2_01_p, path_Et2_01_n, path_Et2_00_p, path_Et2_00_n]
# Sn1list = [path_Sn1_11_p, path_Sn1_11_n, path_Sn1_10_p, path_Sn1_10_n, path_Sn1_01_p, path_Sn1_01_n, path_Sn1_00_p, path_Sn1_00_n]
# Sn2list = [path_Sn2_11_p, path_Sn2_11_n, path_Sn2_10_p, path_Sn2_10_n, path_Sn2_01_p, path_Sn2_01_n, path_Sn2_00_p, path_Sn2_00_n]
# Sp1list = [path_Sp1_11_p, path_Sp1_11_n, path_Sp1_10_p, path_Sp1_10_n, path_Sp1_01_p, path_Sp1_01_n, path_Sp1_00_p, path_Sp1_00_n]
# Sp2list = [path_Sp2_11_p, path_Sp2_11_n, path_Sp2_10_p, path_Sp2_10_n, path_Sp2_01_p, path_Sp2_01_n, path_Sp2_00_p, path_Sp2_00_n]
# p.
Et1list = [path_Et1_11_p, path_Et1_10_p, path_Et1_01_p, path_Et1_00_p]
Et2list = [path_Et2_11_p, path_Et2_10_p, path_Et2_01_p, path_Et2_00_p]
Sn1list = [path_Sn1_11_p,  path_Sn1_10_p,path_Sn1_01_p, path_Sn1_00_p]
Sn2list = [path_Sn2_11_p, path_Sn2_10_p, path_Sn2_01_p, path_Sn2_00_p]
Sp1list = [path_Sp1_11_p, path_Sp1_10_p, path_Sp1_01_p, path_Sp1_00_p]
Sp2list = [path_Sp2_11_p, path_Sp2_10_p, path_Sp2_01_p, path_Sp2_00_p]
k1list = [path_k1_11_p, path_k1_10_p, path_k1_01_p, path_k1_00_p]
k2list = [path_k2_11_p, path_k2_10_p, path_k2_01_p, path_k2_00_p]

# define the font family
font_family = 'Calibri'

filetnamelist = ['Et1', 'Sn1', 'Sp1', 'Et2', 'Sn2', 'Sp2']
title1 = r'$\mathit{E}_{\mathrm{t1}}$' + ' (eV)'
# title1 = r'Primary T$_{\rm eff}$'
title2 = r'$\mathit{E}_{\mathrm{t2}}$' + ' (eV)'
title3 = r'log$(\sigma_{\rm n1})$ ' + r'(cm$^{-3}$)'
title4 = r'log$(\sigma_{\rm n2})$ ' + r'(cm$^{-3}$)'
title5 = r'log$(\sigma_{\rm p1})$ ' + r'(cm$^{-3}$)'
title6 = r'log$(\sigma_{\rm p2})$ ' + r'(cm$^{-3}$)'
title7 = 'log$(k_1)$'
title8 = 'log$(k_2)$'
textlist = ['(a)', '(d)', '(b)', '(e)', '(c)', '(f)', '(g)', '(h)']
titlelist = [title1, title3, title5, title2, title4, title6]
xlabels = []
ylabels = []
titlelist2 = []
for title in titlelist:
    xlabels.append('True ' + title)
    ylabels.append('Predicted ' + title)

title1 = r'E$_{\rm t1}$' + ' (eV)'
title2 = r'E$_{t2}$' + ' (eV)'
title3 = 'log$(\sigma_{n1})$ ' + r'(cm$^{-3}$)'
title4 = 'log$(\sigma_{n2})$ ' + r'(cm$^{-3}$)'
title5 = 'log$(\sigma_{p1})$ ' + r'(cm$^{-3}$)'
title6 = 'log$(\sigma_{p2})$ ' + r'(cm$^{-3}$)'
# title7 = 'log(k$_1)$'
# title8 = 'log(k$_2)$'
titlelist2 = [title1, title3, title5, title2, title4, title6] # , title7, title8]
titlelist = []
for title in titlelist2:
    titlelist.append('RF ' + title + ' prediction')

counter = 0
for task in [Et1list, Sn1list, Sp1list, Et2list, Sn2list, Sp2list]: # k1list, k2list]:
    # extract the dataset.
    filename = filetnamelist[counter]
    Truelist = []
    predictionlist = []
    for path in task:
        data = pd.read_csv(path)
        # the second column is true value:
        true = np.array(data)[:, 1]
        Truelist.append(true)
        # the third column is the prediction value:
        prediction = np.array(data)[:, 2]
        predictionlist.append(prediction)

    # flattern the list into 1d array.
    Truelist = np.array(Truelist).flatten()
    predictionlist = np.array(predictionlist).flatten()

    # subsampling.
    sampleindex = np.random.randint(0, np.shape(Truelist)[0], 10000)
    Truelist = Truelist[sampleindex]
    predictionlist = predictionlist[sampleindex]

    # calculate evaluation matrix.
    R2 = round(r2_score(Truelist, predictionlist), 3)
    R2 = f'{R2:.3f}'
    # print(R2)
    MAE = round(mean_absolute_error(Truelist, predictionlist), 3)
    MAE = f'{MAE:.3f}'
    mape = np.mean(np.abs((np.array(Truelist) - np.array(predictionlist)) / Truelist)) * 100
    # print(MAE)
    print(mape)

    # plotting without centre line.
    fig= plt.figure(facecolor='white', figsize=(5, 5))
    ax = fig.add_subplot(111)
    true = Truelist
    prediction = predictionlist
    plt.scatter(true, prediction, label=('R$^2$' + '=' + str(R2)) + ('; MAE' + '=' + str(MAE)), alpha=0.01, color='green')
    # plt.plot(true, true, color='r')
    plt.xlabel(xlabels[counter], fontsize=22, font=font_family)
    plt.ylabel(ylabels[counter], fontsize=22, font=font_family)
    ax.set_aspect("equal")
    # # plt.title(str(titlelist[counter]), fontsize=25)
    font = font_manager.FontProperties(family=font_family, style='normal', size=20)
    plt.legend(loc=4, framealpha=0.1, prop=font)
    plt.xticks(fontsize=22, font=font_family)
    plt.yticks(fontsize=22, font=font_family)
    # plt.text(0.05, 0.9, textlist[counter], transform=ax.transAxes, fontsize=22, font=font_family)
    if filename[0] == 'S':
        plt.xticks(range(-17, -12))
        plt.yticks(range(-17, -12))
    if filename[0] == 'E':
        plt.xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])
        plt.yticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])
        # plt.xticks([0, 0.2, 0.4, 0.6])
        # plt.yticks([0, 0.2, 0.4, 0.6])
        # plt.xticks([-0.6, -0.4, -0.2, 0])
        # plt.yticks([-0.6, -0.4, -0.2, 0])

    plt.savefig(fname=str(filetnamelist[counter]) + 'without center line' + '.png', bbox_inches='tight')
    plt.show()

    counter = counter + 1

# %% The figure 5 using subplot

# Set 11 p type.
path_Et1_11_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\Et_eV_1set11_800k.csv'
path_Et2_11_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\Et_eV_2set11_800k.csv'
path_Sn1_11_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\logSn_1set11_800k.csv'
path_Sn2_11_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\logSn_2set11_800k.csv'
path_Sp1_11_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\logSp_1set11_800k.csv'
path_Sp2_11_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\logSp_2set11_800k.csv'
path_k1_11_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\logk_1set11_800k.csv'
path_k2_11_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\logk_2set11_800k.csv'
# Set 11 n type.
path_Et1_11_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\Et_eV_1set11_800k_n.csv'
path_Et2_11_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\Et_eV_2set11_800k_n.csv'
path_Sn1_11_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\logSn_1set11_800k_n.csv'
path_Sn2_11_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\logSn_2set11_800k_n.csv'
path_Sp1_11_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\logSp_1set11_800k_n.csv'
path_Sp2_11_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\logSp_2set11_800k_n.csv'
path_k1_11_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\logk_1set11_800k_n.csv'
path_k2_11_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\logk_2set11_800k_n.csv'
# Set 10 p type.
path_Et1_10_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\Re__ML_finish_training\Et_eV_12022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path_Et2_10_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\Re__ML_finish_training\Et_eV_22022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path_Sn1_10_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\Re__ML_finish_training\logSn_12022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path_Sn2_10_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\Re__ML_finish_training\logSn_22022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path_Sp1_10_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\Re__ML_finish_training\logSp_12022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path_Sp2_10_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\Re__ML_finish_training\logSp_22022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path_k1_10_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\logk_12022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path_k2_10_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\logk_22022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
# Set 10 n type.
path_Et1_10_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\Re__\Et_eV_1set10_800k_n.csv'
path_Et2_10_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\Re__\Et_eV_2set10_800k_n.csv'
path_Sn1_10_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\Re__\logSn_1set10_800k_n.csv'
path_Sn2_10_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\Re__\logSn_2set10_800k_n.csv'
path_Sp1_10_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\Re__\logSp_1set10_800k_n.csv'
path_Sp2_10_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\Re__\logSp_2set10_800k_n.csv'
path_k1_10_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\logk_1set10_800k_n.csv'
path_k2_10_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\logk_2set10_800k_n.csv'
# Set 01 p type.
path_Et1_01_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\Et_eV_1set01_800k_p.csv'
path_Et2_01_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\Et_eV_2set01_800k_p.csv'
path_k1_01_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\logk_1set01_800k_p.csv'
path_k2_01_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\logk_2set01_800k_p.csv'
path_Sn1_01_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\logSn_1set01_800k_p.csv'
path_Sn2_01_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\logSn_2set01_800k_p.csv'
path_Sp1_01_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\logSp_1set01_800k_p.csv'
path_Sp2_01_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\logSp_2set01_800k_p.csv'
# Set 01 n type.
path_Et1_01_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\n\set01_n_800k\Et_eV_1set01_800k_n.csv'
path_Et2_01_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\n\set01_n_800k\Et_eV_2set01_800k_n.csv'
path_Sn1_01_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\n\set01_n_800k\logSn_1set01_800k_n.csv'
path_Sn2_01_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\n\set01_n_800k\logSn_2set01_800k_n.csv'
path_Sp1_01_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\n\set01_n_800k\logSp_1set01_800k_n.csv'
path_Sp2_01_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\n\set01_n_800k\logSp_2set01_800k_n.csv'
path_k1_01_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\n\logk_1set01_800k_n.csv'
path_k2_01_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\n\logk_2set01_800k_n.csv'
# Set 00 p type.
path_Et1_00_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\Et_eV_1set00_800k.csv'
path_Et2_00_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\Et_eV_2set00_800k.csv'
path_k1_00_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logk_1set00_800k.csv'
path_k2_00_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logk_2set00_800k.csv'
path_Sn1_00_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logSn_1set00_800k.csv'
path_Sn2_00_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logSn_2set00_800k.csv'
path_Sp1_00_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logSp_1set00_800k.csv'
path_Sp2_00_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logSp_2set00_800k.csv'
# Set 00 n type.
path_Et1_00_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\Et_eV_1set00_n_800k.csv'
path_Et2_00_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\Et_eV_2set00_n_800k.csv'
path_k1_00_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\logk_1set00_n_800k.csv'
path_k2_00_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\logk_2set00_n_800k.csv'
path_Sn1_00_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\logSn_1set00_n_800k.csv'
path_Sn2_00_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\logSn_2set00_n_800k.csv'
path_Sp1_00_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\logSp_1set00_n_800k.csv'
path_Sp2_00_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\logSp_2set00_n_800k.csv'
# put everything in a list:
# both n and p.
# Et1list = [path_Et1_11_p, path_Et1_11_n, path_Et1_10_p, path_Et1_10_n, path_Et1_01_p, path_Et1_01_n, path_Et1_00_p, path_Et1_00_n]
# Et2list = [path_Et2_11_p, path_Et2_11_n, path_Et2_10_p, path_Et2_10_n, path_Et2_01_p, path_Et2_01_n, path_Et2_00_p, path_Et2_00_n]
# Sn1list = [path_Sn1_11_p, path_Sn1_11_n, path_Sn1_10_p, path_Sn1_10_n, path_Sn1_01_p, path_Sn1_01_n, path_Sn1_00_p, path_Sn1_00_n]
# Sn2list = [path_Sn2_11_p, path_Sn2_11_n, path_Sn2_10_p, path_Sn2_10_n, path_Sn2_01_p, path_Sn2_01_n, path_Sn2_00_p, path_Sn2_00_n]
# Sp1list = [path_Sp1_11_p, path_Sp1_11_n, path_Sp1_10_p, path_Sp1_10_n, path_Sp1_01_p, path_Sp1_01_n, path_Sp1_00_p, path_Sp1_00_n]
# Sp2list = [path_Sp2_11_p, path_Sp2_11_n, path_Sp2_10_p, path_Sp2_10_n, path_Sp2_01_p, path_Sp2_01_n, path_Sp2_00_p, path_Sp2_00_n]
# p.
Et1list = [path_Et1_11_p, path_Et1_10_p, path_Et1_01_p, path_Et1_00_p]
Et2list = [path_Et2_11_p, path_Et2_10_p, path_Et2_01_p, path_Et2_00_p]
Sn1list = [path_Sn1_11_p,  path_Sn1_10_p,path_Sn1_01_p, path_Sn1_00_p]
Sn2list = [path_Sn2_11_p, path_Sn2_10_p, path_Sn2_01_p, path_Sn2_00_p]
Sp1list = [path_Sp1_11_p, path_Sp1_10_p, path_Sp1_01_p, path_Sp1_00_p]
Sp2list = [path_Sp2_11_p, path_Sp2_10_p, path_Sp2_01_p, path_Sp2_00_p]
k1list = [path_k1_11_p, path_k1_10_p, path_k1_01_p, path_k1_00_p]
k2list = [path_k2_11_p, path_k2_10_p, path_k2_01_p, path_k2_00_p]

# define the font family
font_family = 'Calibri'

filetnamelist = ['Et1', 'Sn1', 'Sp1', 'Et2', 'Sn2', 'Sp2']
title1 = r'$\mathit{E}_{\mathrm{t1}}$' + ' (eV)'
# title1 = r'Primary T$_{\rm eff}$'
title2 = r'$\mathit{E}_{\mathrm{t2}}$' + ' (eV)'
title3 = r'log$(\sigma_{\rm n1})$ ' + r'(cm$^{-3}$)'
title4 = r'log$(\sigma_{\rm n2})$ ' + r'(cm$^{-3}$)'
title5 = r'log$(\sigma_{\rm p1})$ ' + r'(cm$^{-3}$)'
title6 = r'log$(\sigma_{\rm p2})$ ' + r'(cm$^{-3}$)'
title7 = 'log$(k_1)$'
title8 = 'log$(k_2)$'
textlist = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
titlelist = [title1, title3, title5, title2, title4, title6]
xlabels = []
ylabels = []
titlelist2 = []
for title in titlelist:
    xlabels.append('True ' + title)
    ylabels.append('Predicted ' + title)

title1 = r'E$_{\rm t1}$' + ' (eV)'
title2 = r'E$_{t2}$' + ' (eV)'
title3 = 'log$(\sigma_{n1})$ ' + r'(cm$^{-3}$)'
title4 = 'log$(\sigma_{n2})$ ' + r'(cm$^{-3}$)'
title5 = 'log$(\sigma_{p1})$ ' + r'(cm$^{-3}$)'
title6 = 'log$(\sigma_{p2})$ ' + r'(cm$^{-3}$)'
# title7 = 'log(k$_1)$'
# title8 = 'log(k$_2)$'
titlelist2 = [title1, title3, title5, title2, title4, title6] # , title7, title8]
titlelist = []
for title in titlelist2:
    titlelist.append('RF ' + title + ' prediction')

counter = 0

# create subplots grid
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# fig.subplots_adjust(hspace=0.4)

# define the subplots index
subplot_index = []

# create the figure
fig= plt.figure(figsize=(20, 20))
plt.subplots_adjust(wspace=0.5, hspace=-0.4)

for task in [Et1list, Sn1list, Sp1list, Et2list, Sn2list, Sp2list]: # k1list, k2list]:
    # extract the dataset.
    filename = filetnamelist[counter]
    Truelist = []
    predictionlist = []
    for path in task:
        data = pd.read_csv(path)
        # the second column is true value:
        true = np.array(data)[:, 1]
        Truelist.append(true)
        # the third column is the prediction value:
        prediction = np.array(data)[:, 2]
        predictionlist.append(prediction)

    # flattern the list into 1d array.
    Truelist = np.array(Truelist).flatten()
    predictionlist = np.array(predictionlist).flatten()

    # subsampling.
    sampleindex = np.random.randint(0, np.shape(Truelist)[0], 10000)
    Truelist = Truelist[sampleindex]
    predictionlist = predictionlist[sampleindex]

    # calculate evaluation matrix.
    R2 = round(r2_score(Truelist, predictionlist), 3)
    R2 = f'{R2:.3f}'
    # print(R2)
    MAE = round(mean_absolute_error(Truelist, predictionlist), 3)
    MAE = f'{MAE:.3f}'
    mape = np.mean(np.abs((np.array(Truelist) - np.array(predictionlist)) / Truelist)) * 100
    # print(MAE)
    print(mape)

    # plotting without centre line.
    subplot_str = "2" + "3" + str(counter + 1)
    ax = fig.add_subplot(int(subplot_str))
    true = Truelist
    prediction = predictionlist
    ax.scatter(true, prediction, label=('R$^2$' + '=' + str(R2)) + ('; MAE' + '=' + str(MAE)), alpha=0.01, color='green')
    # plt.plot(true, true, color='r')
    plt.xlabel(xlabels[counter], fontsize=22, font=font_family)
    plt.ylabel(ylabels[counter], fontsize=22, font=font_family)
    ax.set_aspect("equal")
    # # plt.title(str(titlelist[counter]), fontsize=25)
    font = font_manager.FontProperties(family=font_family, style='normal', size=20)
    plt.legend(loc=4, framealpha=0.1, prop=font)
    plt.xticks(fontsize=22, font=font_family)
    plt.yticks(fontsize=22, font=font_family)
    plt.text(0.05, 0.9, textlist[counter], transform=ax.transAxes, fontsize=22, font=font_family)
    if filename[0] == 'S':
        plt.xticks(range(-17, -12))
        plt.yticks(range(-17, -12))
    if filename[0] == 'E':
        plt.xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])
        plt.yticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])
        # plt.xticks([0, 0.2, 0.4, 0.6])
        # plt.yticks([0, 0.2, 0.4, 0.6])
        # plt.xticks([-0.6, -0.4, -0.2, 0])
        # plt.yticks([-0.6, -0.4, -0.2, 0])

    # plt.savefig(fname=str(filetnamelist[counter]) + 'without center line' + '.png', bbox_inches='tight')


    counter = counter + 1

plt.show()

# %%
fig = plt.figure()
fig.add_subplot(231)
plt.scatter([1, 2, 3], [2, 4, 2])
fig.add_subplot(232)
fig.add_subplot(233)
fig.add_subplot(234)
fig.add_subplot(235)
fig.add_subplot(236)
plt.show()