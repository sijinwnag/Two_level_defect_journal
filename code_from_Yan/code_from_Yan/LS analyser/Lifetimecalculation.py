import numpy as np
from semiconductor.recombination.intrinsic import Intrinsic, Radiative, Auger
from semiconductor.recombination.extrinsic import SRH
from semiconductor.material.thermal_velocity import ThermalVelocity as Vel_th



class IDLS():

    _cal_dts = {
        'Ndop' : None,
        'Temp' : None,
        'Doptype' : None,
        'nxc' : None,
        'Et1' : None,
        'tauminor1' : None,
        'k1' : None,
        'Et2' : None,
        'tauminor2' : None,
        'k2' : None
    }

    def __init__(self, **kwarg):

      if kwargs:
        items = [i for i in kwargs.keys() if i in self._cal_dts.keys()]
        for item in items:
            self._cal_dts[item] = kwargs[item]

    def Rad(self, **kwargs):
        if self._cal_dts['Doptype'] == 'n':
            res = Radiative(temp=self._cal_dts['Temp'], Nd=Nidop, Na=1).tau(
                nxc=self._cal_dts['nxc'])
        elif self._cal_dts['Doptype'] == 'p':
            res = Radiative(temp=self._cal_dts['Temp'], Na=Nidop, Nd=1).tau(
                nxc=self._cal_dts['nxc'])
        return res

    def Aug(self, **kwargs):
        if self._cal_dts['Doptype'] == 'n':
            res = Auger(temp=self._cal_dts['Temp'], Nd=Nidop, Na=0).tau(
                nxc=self._cal_dts['nxc'])
        elif self._cal_dts['Doptype'] == 'p':
            res = Auger(temp=self._cal_dts['Temp'], Na=Nidop, Nd=0).tau(
                nxc=self._cal_dts['nxc'])
        return res

    def Intr(self, **kwargs):
        res = 1 / (1. / self.Rad(**kwargs) + 1. / self.Aug(**kwargs))
        return res

    def SRH1(self, **kwargs):
        vth_e, vth_h = Vel_th(temp=self._cal_dts['Temp'])
        if self._cal_dts['Doptype'] == 'n':
            res = SRH(temp=self._cal_dts['Temp'], Nd=Nidop, Na=0)._tau(
                nxc=self._cal_dts['nxc'], tau_e=self._cal_dts['tauminor1'] / self._cal_dts['k1'] * vth_h / vth_e, tau_h=self._cal_dts['tauminor1'], Et=self._cal_dts['Et1'])
        elif self._cal_dts['Doptype'] == 'p':
            res = SRH(temp=self._cal_dts['Temp'], Nd=0, Na=Nidop)._tau(
                nxc=self._cal_dts['nxc'], tau_e=self._cal_dts['tauminor1'], tau_h=self._cal_dts['tauminor1'] * self._cal_dts['k1'] / vth_h * vth_e, Et=self._cal_dts['Et1'])
        return res

    def SRH2(self, **kwargs):
        vth_e, vth_h = Vel_th(temp=self._cal_dts['Temp'])
        if self._cal_dts['Doptype'] == 'n':
            res = SRH(temp=self._cal_dts['Temp'], Nd=Nidop, Na=0)._tau(
                nxc=self._cal_dts['nxc'], tau_e=self._cal_dts['tauminor2'] / self._cal_dts['k2'] * vth_h / vth_e, tau_h=self._cal_dts['tauminor2'], Et=self._cal_dts['Et2'])
        elif self._cal_dts['Doptype'] == 'p':
            res = SRH(temp=self._cal_dts['Temp'], Nd=0, Na=Nidop)._tau(
                nxc=self._cal_dts['nxc'], tau_e=self._cal_dts['tauminor2'], tau_h=self._cal_dts['tauminor2'] * self._cal_dts['k2'] / vth_h * vth_e, Et=self._cal_dts['Et2'])
        return res

    def Eff(self, **kwargs):
        res = 1./(1./self.Intr(**kwargs)+1./self.SRH1(**kwargs)+1./self.SRH2(**kwargs))
        return res
