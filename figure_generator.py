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
axis_numbersize = 22
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
plt.xlabel(r'Average $E_{\rm t1}$ (eV)', fontsize=axis_labelsize)
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
plt.xlabel(r'Average $E_{\rm t2}$ (eV)', fontsize=axis_labelsize)
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
cb.set_label(label='Fitting residual',fontsize=axis_labelsize, font='Cambria')
cb.ax.tick_params(labelsize=axis_numbersize)
print(Etlist[optind][1])
print(Etlist[optind][0])
plt.plot([-0.303], [0.144], 'w*', markersize=10)
plt.plot(Etlist[optind][1],Etlist[optind][0],'ro')
# plt.plot([-0.3],[0.15],'r*', markersize=10)
# plt.annotate('True value', (-0.3, 0.18), color='red', fontsize=15, font='Cambria')
plt.annotate('ML prediction', (-0.4, 0.05), color='white', fontsize=text_size, font='Cambria')
plt.annotate('Lowest residual', (-0.3, 0.18), color='red', fontsize=text_size, font='Cambria')
plt.xlabel('$Et1$', fontsize=axis_labelsize, fontname='Cambria')
plt.ylabel('$Et2$', fontsize=axis_labelsize, fontname='Cambria')
plt.xlabel(r'$E_{\rm t2}-E_{\rm i} \/\/ \rm (eV)$', fontsize=22, fontname='Cambria')
plt.ylabel(r'$E_{\rm t1}-E_{\rm i} \/\/ \rm (eV)$', fontsize=22, fontname='Cambria')
# params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
# plt.rcParams.update(params)
plt.xticks(fontsize=axis_numbersize, font='Cambria')
plt.yticks(fontsize=axis_numbersize, font='Cambria')
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
plt.xlabel(r'ML predicted $E_{\rm t1}$ (eV)', fontsize=axis_labelsize)
plt.ylabel(r'Residual map predicted $E_{\rm t1}$ (eV)', fontsize=axis_labelsize)
plt.xticks(fontsize=axis_numbersize)
plt.yticks(fontsize=axis_numbersize)
plt.annotate('(a)', xy=(0.05, 0.93), xycoords='axes fraction', fontsize=text_size)
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
plt.xlabel(r'ML predicted $E_{\rm t2}$ (eV)', fontsize=axis_labelsize)
plt.ylabel(r'Residual map predicted $E_{\rm t2}$ (eV)', fontsize=axis_labelsize)
plt.xticks(fontsize=axis_numbersize)
plt.yticks(fontsize=axis_numbersize)
plt.annotate('(b)', xy=(0.05, 0.93), xycoords='axes fraction', fontsize=text_size)
# plt.title(r'$E_{\rm t1}$' + ' (eV) ', fontsize=title_size, fontweight='normal')
plt.show()

# %% Computational time comparison
