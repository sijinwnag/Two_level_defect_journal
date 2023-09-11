# %% imports
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.rcParams['text.usetex'] = True
import numpy as np
# from IPython.display import display, Math
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.font_manager as font_manager
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

# %% Plot the defect classification effect from Et1 and Et2

# define the inputs
# x values for Et1 and Et2
x = np.arange(-0.5, 0.6, 0.1)
# y values for average F1 score based on Et1
y_acc_1 = [0.62, 0.76, 0.82, 0.82, 0.82, 0.83, 0.84, 0.84, 0.87, 0.88, 0.89]
y_f1_1 = [0.61, 0.76, 0.82, 0.82, 0.82, 0.83, 0.83, 0.84, 0.87, 0.88, 0.89]
y_err_1 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
# y values for average F1 score based on Et2
y_acc_2 = [0.89, 0.9, 0.89, 0.83, 0.81, 0.81, 0.81, 0.81, 0.81, 0.77, 0.66]
y_f1_2 = [0.89, 0.9, 0.89, 0.83, 0.81, 0.81, 0.81, 0.81, 0.8, 0.77, 0.65]
y_err_2 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
# the visualization data
# Load the data from Yan's code export
Et1_one_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et1_dominate\Et1_alone.csv'
Et1_two_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et1_dominate\two_level.csv'
Et2_one_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_dominate\Et2_one_level_defect.csv'
Et2_two_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_dominate\two_level_defect.csv'
Et2_0_1_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et2_0_1.csv'
Et2_0_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et2_0.csv'
Et12_01_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et12_both_01.csv'
Et12_n_01_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et12_both_n_01.csv'
Et1_01_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et1_0_1.csv'
Et1_n_01_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et1_n_0_1.csv'
# convert to pd dataframe
Et1_one_lifetime = pd.read_csv(Et1_one_path)
Et1_two_lifetime = pd.read_csv(Et1_two_path)
Et2_one_lifetime = pd.read_csv(Et2_one_path)
Et2_two_lifetime = pd.read_csv(Et2_two_path)
Et2_0_1_lifetime = pd.read_csv(Et2_0_1_path)
Et2_0_lifetime = pd.read_csv(Et2_0_path)
Et12_01 = pd.read_csv(Et12_01_path)
Et12_n_01 = pd.read_csv(Et12_n_01_path)
Et1_01 = pd.read_csv(Et1_01_path)
Et1_n_01 = pd.read_csv(Et1_n_01_path)

# define the font
plt.rcParams['font.family'] = fonttype
plt.rc('font', family=fonttype)

# define the figure size
fig = plt.figure(figsize=subplots_figuresize)
fig.subplots_adjust(hspace=0.4, wspace=0.4)

# start the subplot: the first plot of 2 by 2 subplots
plt.subplot(2, 2, 1)
plt.scatter(x, y_f1_1)
# plt.plot(x, y_f1, color='orange')
plt.errorbar(x, y_f1_1, yerr=y_err_1)
plt.xlabel(r'Average $E_{\rm t1}$ (eV)', fontsize=xlabel_size)
plt.ylabel('F1-score', fontsize=xlabel_size)
plt.xticks(fontsize=axis_numbersize)
plt.yticks([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], fontsize=axis_numbersize)
plt.annotate('(a)', xy=chart_number_position, xycoords='axes fraction', fontsize=chart_number_size)

plt.subplot(2, 2, 2)
plt.scatter(x, y_f1_2)
# plt.plot(x, y_f1, color='orange')
plt.errorbar(x, y_f1_2, yerr=y_err_2)
plt.xlabel(r'Average $E_{\rm t2}$ (eV)', fontsize=xlabel_size)
plt.ylabel('F1-score', fontsize=xlabel_size)
plt.xticks(fontsize=axis_numbersize)
plt.yticks([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], fontsize=axis_numbersize)
plt.annotate('(b)', xy=chart_number_position, xycoords='axes fraction', fontsize=chart_number_size)

plt.subplot(2, 2, 4)
plt.plot(Et1_one_lifetime.iloc[:, 0], Et1_one_lifetime.iloc[:, 1], label='One level defect of $E_{t1}$', color='blue', alpha=0.5)
plt.plot(Et1_two_lifetime.iloc[:, 0], Et1_two_lifetime.iloc[:, 1], label='Two level defect', linestyle='dashed', color='red', alpha=0.5)
print(np.max(Et1_one_lifetime.iloc[:, 1]-Et1_two_lifetime.iloc[:, 1]))
plt.legend(fontsize=legend_size, loc='lower left')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Excess carrer concentration ($\rm cm^{-3}$)', fontsize=xlabel_size)
plt.ylabel('Lifetime (s)', fontsize=ylabel_size)
plt.xticks(fontsize=xtick_size)
plt.yticks([1e-5, 1e-4, 1e-3, 10**(-2.5)], fontsize=ytick_size)
plt.annotate('(d)', xy=chart_number_position, xycoords='axes fraction', fontsize=chart_number_size)

plt.subplot(2, 2, 3)
plt.plot(Et2_one_lifetime.iloc[:, 0], Et2_one_lifetime.iloc[:, 1], label='One level defect of $E_{t2}$', color='blue', alpha=0.5)
plt.plot(Et2_two_lifetime.iloc[:, 0], Et2_two_lifetime.iloc[:, 1], label='Two level defect', linestyle='dashed', color='red', alpha=0.5)
print(np.max(Et2_one_lifetime.iloc[:, 1]-Et2_two_lifetime.iloc[:, 1]))
plt.legend(fontsize=legend_size, loc='lower left')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Excess carrer concentration ($\rm cm^{-3}$)', fontsize=xlabel_size)
plt.ylabel('Lifetime (s)', fontsize=ylabel_size)
plt.xticks(fontsize=xtick_size)
plt.yticks([1e-5, 1e-4, 1e-3, 10**(-2.5)], fontsize=ytick_size)
plt.annotate('(c)', xy=chart_number_position, xycoords='axes fraction', fontsize=chart_number_size)
# plt.title(r'$E_{\rm t1}=0.5 \rm eV$; $E_{\rm t2}=0.5 \rm eV$', fontsize = title_size)

plt.show()
plt.savefig('Figure_1.jpg')

# %%
plt.figure()
plt.plot(Et2_one_lifetime.iloc[:, 0], Et2_one_lifetime.iloc[:, 1], label='One level defect with $E_{t1}$ only', color='blue', alpha=0.5)
plt.plot(Et2_two_lifetime.iloc[:, 0], Et2_two_lifetime.iloc[:, 1], label='Two level defect', linestyle='dashed', color='red', alpha=0.5)
print(np.max(Et2_one_lifetime.iloc[:, 1]-Et2_two_lifetime.iloc[:, 1]))
plt.legend(fontsize=legend_size, loc='lower left')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Excess carrer concentration ($\rm cm^{-3}$)', fontsize=xlabel_size)
plt.ylabel('Lifetime (s)', fontsize=ylabel_size)
plt.xticks(fontsize=xtick_size)
plt.yticks([1e-5, 1e-4, 1e-3, 10**(-2.5)], fontsize=ytick_size)
# plt.annotate('(c)', xy=chart_number_position, xycoords='axes fraction', fontsize=chart_number_size)
# plt.title(r'$E_{\rm t1}=0.5 \rm eV$; $E_{\rm t2}=0.5 \rm eV$', fontsize = title_size)
plt.show()

# %% Plot Et1 vs F1 score for defect classification
# define the x and y
x = np.arange(-0.5, 0.6, 0.1)
y_acc = [0.62, 0.76, 0.82, 0.82, 0.82, 0.83, 0.84, 0.84, 0.87, 0.88, 0.89]
y_f1 = [0.61, 0.76, 0.82, 0.82, 0.82, 0.83, 0.83, 0.84, 0.87, 0.88, 0.89]
y_err = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

plt.rcParams['font.family'] = fonttype
plt.rc('font', family=fonttype)
plt.figure(figsize=figuresize)
plt.scatter(x, y_f1)
# plt.plot(x, y_f1, color='orange')
plt.errorbar(x, y_f1, yerr=y_err)
plt.xlabel(r'Average $E_{\rm t1}$ (eV)', fontsize=xlabel_size)
plt.ylabel('F1-score', fontsize=xlabel_size)
plt.xticks(fontsize=axis_numbersize)
plt.yticks([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], fontsize=axis_numbersize)
plt.annotate('(a)', xy=chart_number_position, xycoords='axes fraction', fontsize=chart_number_size)
# plt.title(r'$E_{\rm t1}$' + ' (eV) ', fontsize=title_size, fontweight='normal')
plt.show()


# %% Plot Et2 vs F1 score for defect classification
# define the x and y
x = np.arange(-0.5, 0.6, 0.1)
y_acc = [0.89, 0.9, 0.89, 0.83, 0.81, 0.81, 0.81, 0.81, 0.81, 0.77, 0.66]
y_f1 = [0.89, 0.9, 0.89, 0.83, 0.81, 0.81, 0.81, 0.81, 0.8, 0.77, 0.65]
y_err = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

# plotting
plt.rcParams['font.family'] = fonttype
plt.figure(figsize=figuresize)
plt.scatter(x, y_f1)
# plt.plot(x, y_f1, color='orange')
plt.errorbar(x, y_f1, yerr=y_err)
plt.xlabel(r'Average $E_{\rm t2}$ (eV)', fontsize=xlabel_size)
plt.ylabel('F1-score', fontsize=xlabel_size)
plt.xticks(fontsize=axis_numbersize)
plt.yticks([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], fontsize=axis_numbersize)
plt.annotate('(c)', xy=chart_number_position, xycoords='axes fraction', fontsize=chart_number_size)
# plt.title(r'$E_{\rm t1}$' + ' (eV) ', fontsize=title_size, fontweight='normal')
plt.show()


# %% Plot the case when Et1 is small

# Load the data from Yan's code export
Et1_one_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et1_dominate\Et1_alone.csv'
Et1_two_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et1_dominate\two_level.csv'
Et2_one_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_dominate\Et2_one_level_defect.csv'
Et2_two_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_dominate\two_level_defect.csv'
Et2_0_1_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et2_0_1.csv'
Et2_0_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et2_0.csv'
Et12_01_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et12_both_01.csv'
Et12_n_01_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et12_both_n_01.csv'
Et1_01_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et1_0_1.csv'
Et1_n_01_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et1_n_0_1.csv'

Et1_one_lifetime = pd.read_csv(Et1_one_path)
Et1_two_lifetime = pd.read_csv(Et1_two_path)
Et2_one_lifetime = pd.read_csv(Et2_one_path)
Et2_two_lifetime = pd.read_csv(Et2_two_path)
Et2_0_1_lifetime = pd.read_csv(Et2_0_1_path)
Et2_0_lifetime = pd.read_csv(Et2_0_path)
Et12_01 = pd.read_csv(Et12_01_path)
Et12_n_01 = pd.read_csv(Et12_n_01_path)
Et1_01 = pd.read_csv(Et1_01_path)
Et1_n_01 = pd.read_csv(Et1_n_01_path)

# plot the Et1 low case
plt.figure(facecolor='white', figsize=figuresize)
plt.plot(Et1_one_lifetime.iloc[:, 0], Et1_one_lifetime.iloc[:, 1], label='One level defect of $E_{t1}$', color='blue', alpha=0.5)
plt.plot(Et1_two_lifetime.iloc[:, 0], Et1_two_lifetime.iloc[:, 1], label='Two level defect', linestyle='dashed', color='red', alpha=0.5)
print(np.max(Et1_one_lifetime.iloc[:, 1]-Et1_two_lifetime.iloc[:, 1]))
plt.legend(fontsize=legend_size)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Excess carrer concentration ($\rm cm^{-3}$)', fontsize=xlabel_size)
plt.ylabel('Lifetime (s)', fontsize=ylabel_size)
plt.xticks(fontsize=xtick_size)
plt.yticks(fontsize=ytick_size)
plt.annotate('(c)', xy=(0.1, 0.93), xycoords='axes fraction', fontsize=text_size)
# plt.title(r'$E_{\rm t1}=0.5 \rm eV$; $E_{\rm t2}=0.5 \rm eV$', fontsize = title_size)
# plt.xlim([10**(16.9), 1e17])
# plt.ylim([1e-5, 10**-4.6])
# plt.savefig('Et1_dominate.png', bbox_inches='tight')
plt.show()
# print(np.max(Et1_one_lifetime.iloc[:, 1]-Et1_two_lifetime.iloc[:, 1]))
# plt.figure(facecolor='white')
# plt.plot(Et1_one_lifetime.iloc[:, 0], Et1_one_lifetime.iloc[:, 1], label='One level defect lifetime of $E_{t1}$ only')
# plt.plot(Et1_two_lifetime.iloc[:, 0], Et1_two_lifetime.iloc[:, 1], label='Two level defect', linestyle='dashed')
# plt.legend(fontsize=15)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'Excess carrer concentration ($\rm cm^{-3}$)', fontsize=22)
# plt.ylabel('Lifetime (s)', fontsize=22)
# # plt.title('$E_{t1}=0.5 eV$; $E_{t2}=0.5 eV$' )
# plt.xlim([10**(16.9), 1e17])
# plt.ylim([1e-5, 10**-4.6])
# # plt.xticks(fontsize=20)
# # plt.yticks(fontsize=20)
# # plt.savefig('Et1_dominate_zoomin.png', bbox_inches='tight')
# plt.show()

# %% Plot the case when Et2 is large

# Load the data from Yan's code export
Et1_one_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et1_dominate\Et1_alone.csv'
Et1_two_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et1_dominate\two_level.csv'
Et2_one_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_dominate\Et2_one_level_defect.csv'
Et2_two_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_dominate\two_level_defect.csv'
Et2_0_1_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et2_0_1.csv'
Et2_0_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et2_0.csv'
Et12_01_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et12_both_01.csv'
Et12_n_01_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et12_both_n_01.csv'
Et1_01_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et1_0_1.csv'
Et1_n_01_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et1_n_0_1.csv'

Et1_one_lifetime = pd.read_csv(Et1_one_path)
Et1_two_lifetime = pd.read_csv(Et1_two_path)
Et2_one_lifetime = pd.read_csv(Et2_one_path)
Et2_two_lifetime = pd.read_csv(Et2_two_path)
Et2_0_1_lifetime = pd.read_csv(Et2_0_1_path)
Et2_0_lifetime = pd.read_csv(Et2_0_path)
Et12_01 = pd.read_csv(Et12_01_path)
Et12_n_01 = pd.read_csv(Et12_n_01_path)
Et1_01 = pd.read_csv(Et1_01_path)
Et1_n_01 = pd.read_csv(Et1_n_01_path)

# plot the Et1 low case
plt.figure(facecolor='white', figsize=figuresize)
plt.plot(Et2_one_lifetime.iloc[:, 0], Et2_one_lifetime.iloc[:, 1], label='One level defect lifetime of $E_{t2}$', color='blue', alpha=0.5)
plt.plot(Et2_two_lifetime.iloc[:, 0], Et2_two_lifetime.iloc[:, 1], label='Two level defect', linestyle='dashed', color='red', alpha=0.5)
print(np.max(Et2_one_lifetime.iloc[:, 1]-Et2_two_lifetime.iloc[:, 1]))
plt.legend(fontsize=legend_size)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Excess carrer concentration ($\rm cm^{-3}$)', fontsize=xlabel_size)
plt.ylabel('Lifetime (s)', fontsize=ylabel_size)
plt.xticks(fontsize=xtick_size)
plt.yticks(fontsize=ytick_size)
plt.annotate('(c)', xy=(0.1, 0.93), xycoords='axes fraction', fontsize=text_size)
# plt.title(r'$E_{\rm t1}=0.5 \rm eV$; $E_{\rm t2}=0.5 \rm eV$', fontsize = title_size)
# plt.xlim([10**(16.9), 1e17])
# plt.ylim([1e-5, 10**-4.6])
# plt.savefig('Et1_dominate.png', bbox_inches='tight')
plt.show()
# print(np.max(Et1_one_lifetime.iloc[:, 1]-Et1_two_lifetime.iloc[:, 1]))
# plt.figure(facecolor='white')
# plt.plot(Et1_one_lifetime.iloc[:, 0], Et1_one_lifetime.iloc[:, 1], label='One level defect lifetime of $E_{t1}$ only')
# plt.plot(Et1_two_lifetime.iloc[:, 0], Et1_two_lifetime.iloc[:, 1], label='Two level defect', linestyle='dashed')
# plt.legend(fontsize=15)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'Excess carrer concentration ($\rm cm^{-3}$)', fontsize=22)
# plt.ylabel('Lifetime (s)', fontsize=22)
# # plt.title('$E_{t1}=0.5 eV$; $E_{t2}=0.5 eV$' )
# plt.xlim([10**(16.9), 1e17])
# plt.ylim([1e-5, 10**-4.6])
# # plt.xticks(fontsize=20)
# # plt.yticks(fontsize=20)
# # plt.savefig('Et1_dominate_zoomin.png', bbox_inches='tight')
# plt.show()

# %% Plot the residual map
# %%--load the saved data (different T)
taun1list = np.load(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Yan_DPSS_code\taun1list.npy')
Etlist = np.load(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Yan_DPSS_code\Etlist.npy')
Et1list = Etlist[:, 0]
Et2list = Etlist[:, 1]
taun2list = np.load(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Yan_DPSS_code\taun2list.npy')
k1list = np.load(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Yan_DPSS_code\k1list.npy')
k1listr = np.load(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Yan_DPSS_code\k1listr.npy')
k2list = np.load(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Yan_DPSS_code\k2list.npy')
k2listr = np.load(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Yan_DPSS_code\k2listr.npy')
residuallist = np.load(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Yan_DPSS_code\residuallist.npy')
residuallistr = np.load(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Yan_DPSS_code\residuallistr.npy')
taun1listr = np.load(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Yan_DPSS_code\taun1listr.npy')
taun2list = np.load(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Yan_DPSS_code\taun2list.npy')
taun2listr = np.load(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Yan_DPSS_code\taun2listr.npy')
# np.shape(Etlist)

# dataprocess 2
# taun1listr = np.reshape(taun1list,(len(Et1list),len(Et2list)))
# taun2listr = np.reshape(taun2list,(len(Et1list),len(Et2list)))
# k1listr = np.reshape(k1list,(len(Et1list),len(Et2list)))
# k2listr = np.reshape(k2list,(len(Et1list),len(Et2list)))
# residuallistr = np.reshape(residuallist,(len(Et1list),len(Et2list)))
extent = (Et2list[0], Et2list[-1], Et1list[0], Et1list[-1])
optind = np.argmin(residuallist)

plt.figure(num='Resudual', facecolor='white', figsize=figuresize)
im1 = plt.imshow(residuallistr,extent =extent, aspect='equal', origin = 'lower',norm=colors.LogNorm())
cb = plt.colorbar(im1, label='Fitting residual')
cb.set_label(label='Fitting residual',fontsize=xlabel_size, font=font_family)
cb.ax.tick_params(labelsize=axis_numbersize)
print(Etlist[optind][1])
print(Etlist[optind][0])
plt.plot([-0.303], [0.144], 'w*', markersize=10)
plt.plot(Etlist[optind][1],Etlist[optind][0],'ro')
# plt.plot([-0.3],[0.15],'r*', markersize=10)
# plt.annotate('True value', (-0.3, 0.18), color='red', fontsize=15, font=font_family)
plt.annotate('ML prediction', (-0.4, 0.05), color='white', fontsize=text_size, font=font_family)
plt.annotate('Lowest residual', (-0.3, 0.18), color='red', fontsize=text_size, font=font_family)
plt.xlabel('$Et1$', fontsize=xlabel_size, fontname=font_family)
plt.ylabel('$Et2$', fontsize=xlabel_size, fontname=font_family)
plt.xlabel(r'$E_{\rm t2}-E_{\rm i} \/\/ \rm (eV)$', fontsize=22, fontname=font_family)
plt.ylabel(r'$E_{\rm t1}-E_{\rm i} \/\/ \rm (eV)$', fontsize=22, fontname=font_family)
# params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
# plt.rcParams.update(params)
plt.xticks(fontsize=axis_numbersize, font=font_family)
plt.yticks(fontsize=axis_numbersize, font=font_family)
plt.savefig('residual map low res' + '.png', bbox_inches='tight')
plt.show()

# %% The residual map vs ML prediction

# For Et1
# True_Et1 = np.arange(-0.5, 0.6, 0.1)
# residual_map = True_Et1 + np.random.uniform(-0.01, 0.01, np.shape(True_Et1)) # we know residual map will generate results same as true value but have error based on the Et resolution
# ML = True_Et1 + np.random.uniform(-0.03, 0.03, np.shape(True_Et1))
ML = [-0.49794746, -0.38322031, -0.2900473 , -0.21533488, -0.11027249, 0.02261809,  0.1182406 ,  0.19463504,  0.27850829,  0.38300503, 0.48327495]
residual_map = [-0.50506247, -0.39540019, -0.30869384, -0.20506867, -0.09674902, -0.00246835,  0.10694732,  0.20419474,  0.30046075,  0.39047682, 0.50749813]

# plotting
x = ML
y = residual_map
plt.rcParams['font.family'] = fonttype
plt.rc('font', family=fonttype)
plt.figure(figsize=figuresize)
# plt.scatter(x, y)
# plt.plot(x, y_f1, color='orange')
plt.errorbar(x, y, yerr=0.01, xerr=0.03)
plt.xlabel(r'ML predicted $E_{\rm t1}$ (eV)', fontsize=xlabel_size)
plt.ylabel(r'Residual map predicted $E_{\rm t1}$ (eV)', fontsize=xlabel_size)
plt.xticks(fontsize=axis_numbersize)
plt.yticks(fontsize=axis_numbersize)
plt.annotate('(a)', xy=chart_number_position, xycoords='axes fraction', fontsize=text_size)
# plt.title(r'$E_{\rm t1}$' + ' (eV) ', fontsize=title_size, fontweight='normal')
plt.show()

# For Et2
# True_Et2 = np.arange(-0.5, 0.6, 0.1)
# residual_map = True_Et2 + np.random.uniform(-0.01, 0.01, np.shape(True_Et2)) # we know residual map will generate results same as true value but have error based on the Et resolution
# ML = True_Et2 + np.random.uniform(-0.05, 0.05, np.shape(True_Et2))
ML = [-5.47536548e-01, -3.97000169e-01, -2.72439132e-01, -1.56597562e-01, -9.98478041e-02, -1.83663528e-04,  5.92034895e-02,  2.24919450e-01, 3.38970389e-01,  3.52404289e-01,  4.74637175e-01]
residual_map = [-0.50706917, -0.39232222, -0.30677661, -0.19509884, -0.09890083, -0.0045195 ,  0.0972121 ,  0.19626744,  0.30619249,  0.40383695, 0.50494745]

# plotting
x = ML
y = residual_map
plt.rcParams['font.family'] = fonttype
plt.rc('font', family=fonttype)
plt.figure(figsize=figuresize)
# plt.scatter(x, y)
# plt.plot(x, y_f1, color='orange')
plt.errorbar(x, y, yerr=0.01, xerr=0.03)
plt.xlabel(r'ML predicted $E_{\rm t2}$ (eV)', fontsize=xlabel_size)
plt.ylabel(r'Residual map predicted $E_{\rm t2}$ (eV)', fontsize=xlabel_size)
plt.xticks(fontsize=axis_numbersize)
plt.yticks(fontsize=axis_numbersize)
plt.annotate('(b)', xy=chart_number_position, xycoords='axes fraction', fontsize=text_size)
# plt.title(r'$E_{\rm t1}$' + ' (eV) ', fontsize=title_size, fontweight='normal')
plt.show()

# %% Plot the regression results for 6 parameters
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
# n
# Et1list = [path_Et1_11_n, path_Et1_10_n, path_Et1_01_n, path_Et1_00_n]
# Et2list = [path_Et2_11_n, path_Et2_10_n, path_Et2_01_n, path_Et2_00_n]
# Sn1list = [path_Sn1_11_n,  path_Sn1_10_n,path_Sn1_01_n, path_Sn1_00_n]
# Sn2list = [path_Sn2_11_n, path_Sn2_10_n, path_Sn2_01_n, path_Sn2_00_n]
# Sp1list = [path_Sp1_11_n, path_Sp1_10_n, path_Sp1_01_n, path_Sp1_00_n]
# Sp2list = [path_Sp2_11_n, path_Sp2_10_n, path_Sp2_01_n, path_Sp2_00_n]
# k1list = [path_k1_11_n, path_k1_10_n, path_k1_01_n, path_k1_00_n]
# k2list = [path_k2_11_n, path_k2_10_n, path_k2_01_n, path_k2_00_n]
# p set 11:
# Et1list = [path_Et1_11_p]
# Et2list = [path_Et2_11_p]
# Sn1list = [path_Sn1_11_p]
# Sn2list = [path_Sn2_11_p]
# Sp1list = [path_Sp1_11_p]
# Sp2list = [path_Sp2_11_p]
# k1list = [path_k1_11_p]
# k2list = [path_k2_11_p]

# # Extract the data.
# Truelist = []
# predictionlist = []
# for path in Et1list:
#     data = pd.read_csv(path)
#     # the second column is true value:
#     true = np.array(data)[:, 1]
#     Truelist.append(true)
#     # the third column is the prediction value:
#     prediction = np.array(data)[:, 2]
#     predictionlist.append(prediction)

# # flattern the list into 1d array.
# Truelist = np.array(Truelist).flatten()
# predictionlist = np.array(predictionlist).flatten()

# # subsampling.
# sampleindex = np.random.randint(0, np.shape(Truelist)[0], 10000)
# Truelist = Truelist[sampleindex]
# predictionlist = predictionlist[sampleindex]
# error = np.absolute(Truelist-predictionlist)

# # calculate evaluation matrix.
# R2 = r2_score(Truelist, predictionlist)
# print(R2)
# MAE = mean_absolute_error(Truelist, predictionlist)
# print(MAE)


# # Plotting perfect prediction sinario
# fig= plt.figure(facecolor='white', figsize=(6, 6))
# ax = fig.add_subplot(111)
# true = Truelist
# prediction = predictionlist
# plt.scatter(true, prediction, label=('$R^2$' + '=' + str(round(R2, 3))) + ('; MAE' + '=' + str(round(MAE, 3))), alpha=0, color='green')
# plt.plot(true, true, color='r')
# plt.xlabel('True', fontsize=20)
# plt.ylabel('Prediction', fontsize=20)
# ax.set_aspect("equal")
# # # plt.text(0, 0.5, alphabet[k], fontsize=20)
# # # # plt.title('$E_{t1}$', fontsize=25)
# # # plt.title('True vs prediction plot', fontsize=20)
# # plt.legend(loc=4, framealpha=0.1, fontsize=20)
# plt.xticks(fontsize=15, font=font_family)
# plt.yticks(fontsize=15, font=font_family)
# plt.savefig(fname=str('Et1') + '.png', bbox_inches='tight')
# plt.show()

# define the font family
font_family = 'Calibri'

filetnamelist = ['Et1', 'Et2', 'Sn1', 'Sn2', 'Sp1', 'Sp2', 'k1', 'k2']
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
titlelist = [title1, title2, title3, title4, title5, title6, title7, title8]
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
title7 = 'log(k$_1)$'
title8 = 'log(k$_2)$'
titlelist2 = [title1, title2, title3, title4, title5, title6, title7, title8]
titlelist = []
for title in titlelist2:
    titlelist.append('RF ' + title + ' prediction')

counter = 0
for task in [Et1list, Et2list, Sn1list, Sn2list, Sp1list, Sp2list, k1list, k2list]:
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
        # plt.xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])
        # plt.yticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])
        plt.xticks([0, 0.2, 0.4, 0.6])
        plt.yticks([0, 0.2, 0.4, 0.6])
        # plt.xticks([-0.6, -0.4, -0.2, 0])
        # plt.yticks([-0.6, -0.4, -0.2, 0])

    plt.savefig(fname=str(filetnamelist[counter]) + 'without center line' + '.png', bbox_inches='tight')
    plt.show()

    # # plotting with centre line.
    # fig= plt.figure(facecolor='white', figsize=(5, 5))
    # ax = fig.add_subplot(111)
    # true = Truelist
    # prediction = predictionlist
    # plt.scatter(true, prediction, label=('R$^2$' + '=' + str(round(R2, 3))) + ('; MAE' + '=' + str(round(MAE, 3))), alpha=0.01, color='green')
    # plt.plot(true, true, color='r')
    # plt.xlabel(xlabels[counter], fontsize=22, font=font_family)
    # plt.ylabel(ylabels[counter], fontsize=22, font=font_family)
    # ax.set_aspect("equal")
    # # # plt.text(0, 0.5, alphabet[k], fontsize=20)
    # # # # plt.title(str(titlelist[counter]), fontsize=25)
    # font = font_manager.FontProperties(family=font_family, style='normal', size=20)
    # plt.legend(loc=4, framealpha=0.1, prop=font)
    # plt.xticks(fontsize=15, font=font_family)
    # plt.yticks(fontsize=15, font=font_family)
    # if filename[0] == 'S':
    #     # print(filename)
    #     plt.xticks(range(-17, -12))
    #     plt.yticks(range(-17, -12))
    # if filename[0] == 'E' and len(task)>1:
    #     plt.xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])
    #     plt.yticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])
    # plt.savefig(fname=str(filetnamelist[counter]) + 'with center line' + '.png', bbox_inches='tight')
    # plt.show()

    counter = counter + 1

# %%
# define the path
path1 = r"C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et1_055_Et2_055.csv"
path2 = r"C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et1_055_Et2_050.csv"
path3 = r"C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et1_055_Et2_045.csv"
# read the data
Et2_055_data = pd.read_csv(path1)
Et2_050_data = pd.read_csv(path2)
Et2_045_data = pd.read_csv(path3)
# plot the data
plt.figure()
plt.plot(Et2_055_data.iloc[:, 0], Et2_055_data.iloc[:, 1], label='$E_{t2}$=0.55 eV', color='blue', alpha=0.5)
plt.plot(Et2_050_data.iloc[:, 0], Et2_050_data.iloc[:, 1], label='$E_{t2}$=0.50 eV', linestyle='dashed', color='red', alpha=0.5)
plt.plot(Et2_045_data.iloc[:, 0], Et2_045_data.iloc[:, 1], label='$E_{t2}$=0.45 eV', linestyle='dashed', color='green', alpha=0.5)
# print(np.max(Et2_one_lifetime.iloc[:, 1]-Et2_two_lifetime.iloc[:, 1]))
plt.legend(fontsize=legend_size, loc='lower left')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Excess carrer concentration ($\rm cm^{-3}$)', fontsize=xlabel_size)
plt.ylabel('Lifetime (s)', fontsize=ylabel_size)
plt.xticks(fontsize=xtick_size)
plt.yticks([1e-5, 1e-4, 1e-3, 10**(-2.5)], fontsize=ytick_size)
# plt.annotate('(c)', xy=chart_number_position, xycoords='axes fraction', fontsize=chart_number_size)
# plt.title(r'$E_{\rm t1}=0.5 \rm eV$; $E_{\rm t2}=0.5 \rm eV$', fontsize = title_size)
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
plt.plot(two_level.iloc[:, 0], two_level.iloc[:, 1], label='Two level defect', color='blue', alpha=0.5)
plt.plot(one_level.iloc[:, 0], one_level.iloc[:, 1], label='$E_{t1}$ recombination', linestyle='dashed', color='red', alpha=0.5)
plt.plot(one_level.iloc[:, 0], one_level.iloc[:, 2], label='$E_{t2}$ recombination', linestyle='dashed', color='green', alpha=0.5)
plt.plot(one_level.iloc[:, 0], one_level.iloc[:, 3], label='Two independent defects', linestyle='dashed', color='red', alpha=1)
plt.legend(fontsize=11, loc='upper left', ncol=2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Excess carrer concentration ($\rm cm^{-3}$)', fontsize=xlabel_size)
plt.ylabel('Lifetime (s)', fontsize=ylabel_size)
plt.xticks(fontsize=xtick_size)
plt.yticks([1e-5, 1e-3, 1e-1, 10], fontsize=ytick_size)
plt.ylim((1e-5, 1e3))
plt.show()
# %% Figure 4 (a) in Journal
# define the path
path1 = r"D:\study\thesis_data_storage\journal\set_classification\plot_data\Et_04_01.csv"
path2 = r"D:\study\thesis_data_storage\journal\set_classification\plot_data\Et_04_n01.csv"
path3 = r"D:\study\thesis_data_storage\journal\set_classification\plot_data\Et_n04_01.csv"
path4 = r"D:\study\thesis_data_storage\journal\set_classification\plot_data\Et_n04_n01.csv"
# read the data
data1 = pd.read_csv(path1)
data2 = pd.read_csv(path2)
data3 = pd.read_csv(path3)
data4 = pd.read_csv(path4)

# Plot the data
fig, axis = plt.subplots(1, 2, figsize=(15, 5))

# First subplot
axis[0].plot(data1.iloc[:, 0], data1.iloc[:, 1]*1e6, label='$E_{t1}$=0.4 eV; $E_{t2}$=0.1 eV', color='blue', alpha=0.5)
axis[0].plot(data2.iloc[:, 0], data2.iloc[:, 1]*1e6, label='$E_{t1}$=0.4 eV; $E_{t2}$=-0.1 eV', linestyle='dashed', color='red', alpha=0.5)
axis[0].legend(fontsize=legend_size, loc='lower left', ncol=1)
axis[0].set_xscale('log')
axis[0].set_xlabel(r'Excess carrier concentration ($\rm cm^{-3}$)', fontsize=xlabel_size)
axis[0].set_ylabel('Lifetime (µs)', fontsize=ylabel_size)
axis[0].set_xticks([1e12, 1e13, 1e14, 1e15, 1e16, 1e17], fontsize=xtick_size)
axis[0].set_yticks([1, 2, 3, 4], fontsize=ytick_size)
axis[0].text(0.95, 0.95, '(a)', transform=axis[0].transAxes, fontsize=20, va='top', ha='right')
axis[0].tick_params(axis='both', which='major', labelsize=xtick_size)
# Second subplot
axis[1].plot(data3.iloc[:, 0], data3.iloc[:, 1]*1e6, label='$E_{t1}$=-0.4 eV; $E_{t2}$=0.1 eV', color='blue', alpha=0.5)
axis[1].plot(data4.iloc[:, 0], data4.iloc[:, 1]*1e6, label='$E_{t1}$=-0.4 eV; $E_{t2}$=-0.1 eV', linestyle='dashed', color='red', alpha=0.5)
axis[1].legend(fontsize=legend_size, loc='lower left', ncol=1)
axis[1].set_xscale('log')
axis[1].set_xlabel(r'Excess carrier concentration ($\rm cm^{-3}$)', fontsize=xlabel_size)
axis[1].set_ylabel('Lifetime (µs)', fontsize=ylabel_size)
axis[1].set_xticks([1e12, 1e13, 1e14, 1e15, 1e16, 1e17], fontsize=xtick_size)
axis[1].set_yticks([0.1, 0.5, 1], fontsize=ytick_size)
axis[1].text(0.95, 0.95, '(b)', transform=axis[1].transAxes, fontsize=20, va='top', ha='right')
axis[1].tick_params(axis='both', which='major', labelsize=xtick_size)
plt.show()
