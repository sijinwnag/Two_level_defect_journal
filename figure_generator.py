# %% imports
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.rcParams['text.usetex'] = True
import numpy as np
# from IPython.display import display, Math
import pandas as pd
# %% Scatter plot
'''
Image standard:
fontsize = 22;
Title font size = 22
figure size = 7 by 7;
'''
# define the parameters
figuresize = (7, 7)
axis_labelsize=22
title_size=25
axis_numbersize = 15
text_size=22
fonttype = "Arial"

# %% Plot Et1 vs F1 score for defect classification
# define the x and y
x = np.arange(-0.5, 0.6, 0.1)
y_acc = [0.62, 0.76, 0.82, 0.82, 0.82, 0.83, 0.84, 0.84, 0.87, 0.88, 0.89]
y_f1 = [0.61, 0.76, 0.82, 0.82, 0.82, 0.83, 0.83, 0.84, 0.87, 0.88, 0.89]
y_err = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

# plotting
plt.rcParams['font.family'] = fonttype
plt.rc('font', family=fonttype)
plt.figure(figsize=figuresize)
plt.scatter(x, y_f1)
# plt.plot(x, y_f1, color='orange')
plt.errorbar(x, y_f1, yerr=y_err)
plt.xlabel(r'$E_{\rm t1}$ (eV)', fontsize=axis_labelsize)
plt.ylabel('F1-score', fontsize=axis_labelsize)
plt.xticks(fontsize=axis_numbersize)
plt.yticks([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], fontsize=axis_numbersize)
plt.annotate('(a)', xy=(0.05, 0.93), xycoords='axes fraction', fontsize=text_size)
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
plt.xlabel(r'$E_{\rm t2}$ (eV)', fontsize=axis_labelsize)
plt.ylabel('F1-score', fontsize=axis_labelsize)
plt.xticks(fontsize=axis_numbersize)
plt.yticks([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], fontsize=axis_numbersize)
plt.annotate('(c)', xy=(0.05, 0.93), xycoords='axes fraction', fontsize=text_size)
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
plt.figure(facecolor='white', figsize=(5, 5))
plt.plot(Et1_one_lifetime.iloc[:, 0], Et1_one_lifetime.iloc[:, 1], label='One level defect lifetime of $E_{t1}$')
plt.plot(Et1_two_lifetime.iloc[:, 0], Et1_two_lifetime.iloc[:, 1], label='Two level defect')
print(np.max(Et1_one_lifetime.iloc[:, 1]-Et1_two_lifetime.iloc[:, 1]))
plt.legend(fontsize=13)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Excess carrer concentration ($\rm cm^{-3}$)', fontsize=22)
plt.ylabel('Lifetime (s)', fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.title('$E_{t1}=0.5 eV$; $E_{t2}=0.5 eV$' )
# plt.xlim([10**(16.9), 1e17])
# plt.ylim([1e-5, 10**-4.6])
# plt.savefig('Et1_dominate.png', bbox_inches='tight')
plt.show()
# print(np.max(Et1_one_lifetime.iloc[:, 1]-Et1_two_lifetime.iloc[:, 1]))
plt.figure(facecolor='white')
plt.plot(Et1_one_lifetime.iloc[:, 0], Et1_one_lifetime.iloc[:, 1], label='One level defect lifetime of $E_{t1}$ only')
plt.plot(Et1_two_lifetime.iloc[:, 0], Et1_two_lifetime.iloc[:, 1], label='Two level defect')
plt.legend(fontsize=15)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Excess carrer concentration ($\rm cm^{-3}$)', fontsize=22)
plt.ylabel('Lifetime (s)', fontsize=22)
# plt.title('$E_{t1}=0.5 eV$; $E_{t2}=0.5 eV$' )
plt.xlim([10**(16.9), 1e17])
plt.ylim([1e-5, 10**-4.6])
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.savefig('Et1_dominate_zoomin.png', bbox_inches='tight')
plt.show()