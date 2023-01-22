# %% imports
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.rcParams['text.usetex'] = True
import numpy as np
# from IPython.display import display, Math

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