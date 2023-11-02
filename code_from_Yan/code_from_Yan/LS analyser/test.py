import numpy as np
import matplotlib.pyplot as plt


a = [1, 2, 3]
a = a + ['4']
print(a)
# X = np.random.rand(100, 1000)
# xs = np.mean(X, axis=1)
# ys = np.std(X, axis=1)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('click on point to plot time series')
# line, = ax.plot(xs, ys, 'o', picker=5)  # 5 points tolerance
#
#
# def onpick(event):
#
#     if event.artist != line:
#         return True
#
#     N = len(event.ind)
#     if not N:
#         return True
#
#     figi = plt.figure()
#     for subplotnum, dataind in enumerate(event.ind):
#         ax = figi.add_subplot(N, 1, subplotnum + 1)
#         ax.plot(X[dataind])
#         ax.text(0.05, 0.9, 'mu=%1.3f\nsigma=%1.3f' % (xs[dataind], ys[dataind]),
#                 transform=ax.transAxes, va='top')
#         ax.set_ylim(-0.5, 1.5)
#     figi.show()
#     return True
#
#
# fig.canvas.mpl_connect('pick_event', onpick)
#
# plt.show()

# import numpy as np
# import matplotlib
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt
# import pickle
# import scipy.constants as const
# from semiconductor.material.thermal_velocity import ThermalVelocity as Vel_th
# plt.ion()
# plt.plot([1.6, 2.7])
# matplotlib.pyplot.switch_backend('Qt5Agg')
# import seaborn as sns
# sns.set(color_codes=True)

# fig, ax = plt.subplots()
# ax.plot([1, 2, 3], [10, -10, 30])
# pickle.dump(fig, open('FigureObject.fig.pickle', 'wb'))
# plt.show()

# figx = pickle.load(open('FigureObject.fig.pickle', 'rb'))
# data = figx.axes[0].lines[0].get_data()
# plt.show()
# print(data)
# kb = const.k / const.e
# T = 300
# Nt = np.logspace(10, 20, 100)
# Nt = 1e16
# Et = 0.3
# se = 1e-15
# sp = 1e-19
# ve = 2.05e7
# vp = 1.69e7
# ni = 1e10
# an = se * ve
# ap = sp * vp
# p0 = 1.3e16
# n0 = ni**2 / p0
# n1 = ni * np.exp(Et / kb / T)
# p1 = ni * np.exp(-Et / kb / T)

# tau_i = 1 / (ap * (p0 + p1 + Nt / (1 + p0 / p1)) +
#              an * (n0 + n1 + Nt / (1 + n0 / n1)))
#
# tau_t = ((p0 + p1 + Nt / (1 + p0 / p1)) / an / Nt +
#          (n0 + n1 + Nt / (1 + n0 / n1)) / ap / Nt) / \
#     (n0 + p0 + Nt / (1 + n0 / n1) / (1 + n1 / n0))  # - \
# # 1 / (ap * (p0 + p1 + Nt / (1 + p0 / p1)) +
# # an * (n0 + n1 + Nt / (1 + n0 / n1)))
# tau_t = 1 / (n0 + p0) / ap / (1 + n0 / n1)
#
# tau_SSn = ((p0 + p1) / an / Nt + (n0 + n1 + Nt / (1 + n0 / n1)) /
#            ap / Nt) / (n0 + p0 + Nt / (1 + n0 / n1) / (1 + n1 / n0))
#
# tau_SSp = ((p0 + p1 + Nt / (1 + p0 / p1)) / an / Nt + (n0 + n1) /
#            ap / Nt) / (n0 + p0 + Nt / (1 + n0 / n1) / (1 + n1 / n0))
#
# tau_eff = ((n0 + n1) / Nt / ap + (p0 + p1) / an / Nt) / (n0 + p0)
# Et = np.linspace(-0.6, 0.6)
# p1 = ni * np.exp(-Et / kb / T)
# plt.figure()
# plt.plot(Et, 1 / 65 / vp / (p1 + p0))
# plt.xlabel('Et - Ei [eV]')
# plt.ylabel('sigma_p [cm2]')
# print(1e6 * tau_t)
# print((n0 + n1) / Nt, 1 / (1 + n0 / n1))
# print(n1 / (Nt / (1 + n0 / n1)))
# print(1e6 * tau_i, 1e6 * tau_t, 1e6 * tau_SSn,
#       1e6 * tau_SSp, 1e6 * tau_eff, 1e6 / an / n1)
# print(1e6 / (ap * (p0 + p1) + an * (n0 + n1)))
# print('{:6e}'.format(ve[0]))
# plt.draw()
