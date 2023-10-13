import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QListWidgetItem, QLineEdit, QLabel, QRadioButton, QGridLayout, QPushButton
from PyQt5 import QtGui, QtWidgets
from uiLSProcess import Ui_IDLSanalyzer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np


class rawdata(object):
    def __init__(self, name=None, Ndop=None, temp=None, doptype=None, nxc=None, tau=None, PCPL='PC', plotopt=True, uid=None, **kwarg):
        self.name = name
        self.Ndop = Ndop
        self.temp = temp
        self.doptype = doptype
        self.nxc = nxc
        self.tau = tau
        self.uid = uid
        self.plotopt = plotopt
        self.PCPL = PCPL

    def checkset(self):
        if (self.Ndop is None) or (self.temp is None) or (self.doptype is None):
            return False
        else:
            return True


class LSana(QMainWindow, Ui_IDLSanalyzer):

    def __init__(self, parent=None):
        self.Rawdata = []
        QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.listWidget_data.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        self.listWidget_data.itemSelectionChanged.connect(self.checkselection)

        self.listWidget_plotlist.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        self.listWidget_plotlist.itemSelectionChanged.connect(
            self.checkselection)

        self.listWidget_analysis.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        self.listWidget_analysis.itemSelectionChanged.connect(
            self.checkselection)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.verticalLayout.addWidget(self.canvas)
        self.verticalLayout.addWidget(self.toolbar)
        self.ax1 = self.figure.add_subplot(111)
        self.ax1.set_xlabel(r'Excess carrier density $[cm^{-3}]$')
        self.ax1.set_ylabel('Lifetime [s]')
        self.figure.tight_layout()

        self.pushButton_load.clicked.connect(self.loaddata)

        self.pushButton_del.setEnabled(False)
        self.pushButton_del.clicked.connect(self.deldata)

        self.pushButton_data2plot.clicked.connect(self.add2plot)
        self.pushButton_data2plot.setEnabled(False)

        self.pushButton_delplot.clicked.connect(self.delplot)
        self.pushButton_delplot.setEnabled(False)

        self.pushButton_set.clicked.connect(self.Opensetting)
        self.pushButton_set.setEnabled(False)

        self.pushButton_data2fit.clicked.connect(self.add2analysis)
        self.pushButton_data2fit.clicked.connect(self.checkselection)
        self.pushButton_data2fit.setEnabled(False)

        self.pushButton_delanalysis.clicked.connect(self.delAnalysis)
        self.pushButton_delanalysis.clicked.connect(self.checkselection)
        self.pushButton_delanalysis.setEnabled(False)

        self.comboBox_plotwhat.currentIndexChanged.connect(self.updatedataplot)
        self.comboBox_plotinverse.currentIndexChanged.connect(
            self.updatedataplot)

        self.pushButton_expdata.clicked.connect(self.export)
        self.pushButton_expdata.setEnabled(False)

        self.pushButton_corp.clicked.connect(self.croppreview)
        self.pushButton_corp.setEnabled(False)

        self.pushButton_intcorrect.clicked.connect(self.correctintrinsic)
        self.pushButton_intcorrect.setEnabled(False)

        self.pushButton_sub2data.clicked.connect(self.sub2)
        self.pushButton_sub2data.setEnabled(False)

        self.pushButton_mergetwo.clicked.connect(self.merge2)
        self.pushButton_mergetwo.setEnabled(False)

        self.pushButton_Analysis.clicked.connect(self.StartAnalysis)
        self.pushButton_Analysis.setEnabled(False)

        self.lineEdit_croplow.textChanged[str].connect(self.checkcorp)
        self.lineEdit_crophigh.textChanged[str].connect(self.checkcorp)

        self.pushButton_corpconfirm.setEnabled(False)
        self.pushButton_corpconfirm.clicked.connect(self.crop)

        self.pushButton_corpcancel.setEnabled(False)
        self.pushButton_corpcancel.clicked.connect(self.cropcancel)

    def loaddata(self):
        filename = QFileDialog.getOpenFileNames(
            self, caption='Choose data file', filter="Exported file (*.txt);;Sinton file(*.xlsm)")
        if filename[0] != '':
            for fname in filename[0]:
                if os.path.splitext(fname)[1] == '.txt':
                    print('a')
                    data = np.genfromtxt(fname, delimiter='\t',usecols=np.arange(0,5), names=[
                                         'Time', 'nxcPC', 'nxcPL', 'tauPC', 'tauPL'])
                    nxcPC = data['nxcPC'][1:]
                    nxcPL = data['nxcPL'][1:]
                    tauPC = data['tauPC'][1:]
                    tauPL = data['tauPL'][1:]
                    idx = np.argsort(nxcPC)
                    nxcPC = nxcPC[idx]
                    tauPC = tauPC[idx]
                    idx = np.argsort(nxcPL)
                    nxcPL = nxcPL[idx]
                    tauPL = tauPL[idx]
                    PL = rawdata(
                        name='PL_' + (os.path.splitext(os.path.basename(fname))[0]), nxc=nxcPL, tau=tauPL)
                    PC = rawdata(
                        name='PC_' + (os.path.splitext(os.path.basename(fname))[0]), nxc=nxcPC, tau=tauPC)
                    self.Rawdata.append(PL)
                    self.Rawdata.append(PC)
                self.updateuid()
                self.updateDataList()
                self.updateplotlist()
                self.updatedataplot()

    def updateDataList(self):
        self.listWidget_data.clear()
        for data in self.Rawdata:
            dataitem = QListWidgetItem(parent=self.listWidget_data)
            dataitem.setText(data.name)
            dataitem.setData(32, data.uid)
            self.listWidget_data.addItem(dataitem)
            if data.checkset() is False:
                dataitem.setForeground(QtGui.QBrush(QtGui.QColor(255, 0, 0)))
            if data.checkset() is True:
                dataitem.setForeground(QtGui.QBrush(QtGui.QColor(0, 255, 0)))

    def updateplotlist(self):
        self.listWidget_plotlist.clear()
        for data in self.Rawdata:
            if data.plotopt:
                dataitem = QListWidgetItem(parent=self.listWidget_plotlist)
                dataitem.setText(data.name)
                dataitem.setData(32, data.uid)
                self.listWidget_plotlist.addItem(dataitem)

    def updateuid(self):
        for data in self.Rawdata:
            data.uid = self.Rawdata.index(data)

    def updatedataplot(self):
        self.ax1.clear()
        self.ax1.grid()
        self.ax1.set_xlabel(r'Excess carrier density $[cm^{-3}]$')
        if self.comboBox_plotwhat.currentIndex() == 0:
            for data in self.Rawdata:
                if data.plotopt:
                    if self.comboBox_plotinverse.currentIndex() == 0:
                        self.ax1.set_ylabel('Lifetime [s]')
                        self.ax1.plot(data.nxc, data.tau, '.', label=data.name)
                    elif self.comboBox_plotinverse.currentIndex() == 1:
                        self.ax1.set_ylabel('Inverse Lifetime [s-1]')
                        self.ax1.plot(data.nxc, 1. / data.tau,
                                      '.', label=data.name)
        elif self.comboBox_plotwhat.currentIndex() == 1:
            for i in range(self.listWidget_analysis.count()):
                item = self.listWidget_analysis.item(i)
                for data in self.Rawdata:
                    if data.uid == item.data(32):
                        if self.comboBox_plotinverse.currentIndex() == 0:
                            self.ax1.set_ylabel('Lifetime [s]')
                            self.ax1.plot(data.nxc, data.tau,
                                          '.', label=data.name)
                        elif self.comboBox_plotinverse.currentIndex() == 1:
                            self.ax1.set_ylabel('Inverse Lifetime [s-1]')
                            self.ax1.plot(data.nxc, 1. / data.tau,
                                          '.', label=data.name)
        self.ax1.semilogx()
        self.ax1.legend(loc=0)
        self.canvas.draw()
        self.figure.tight_layout()

    def deldata(self):
        for item in self.listWidget_data.selectedItems():
            for data in self.Rawdata:
                if data.uid == item.data(32):
                    self.Rawdata.remove(data)
        self.updateDataList()
        self.updateuid()
        self.updateplotlist()
        self.updatedataplot()

    def delplot(self):
        for item in self.listWidget_plotlist.selectedItems():
            for data in self.Rawdata:
                if data.uid == item.data(32):
                    data.plotopt = False
        self.updateplotlist()
        self.updatedataplot()

    def add2plot(self):
        for item in self.listWidget_data.selectedItems():
            for data in self.Rawdata:
                if data.uid == item.data(32):
                    data.plotopt = True
        self.updateplotlist()
        self.updatedataplot()

    def Opensetting(self):
        self.dialog = QtWidgets.QDialog()
        grid = QGridLayout(self.dialog)
        self.ntype = QRadioButton('n-type')
        self.ptype = QRadioButton('p-type')
        self.dop = QLineEdit()
        self.temp = QLineEdit()
        Ldop = QLabel('Ndop (cm-3)')
        Ltemp = QLabel('Temp (K)')
        self.ok = QPushButton('OK')
        self.ok.setEnabled(False)
        grid.addWidget(self.ntype, 0, 0)
        grid.addWidget(self.ptype, 0, 1)
        grid.addWidget(Ldop, 1, 0)
        grid.addWidget(self.dop, 1, 1)
        grid.addWidget(Ltemp, 2, 0)
        grid.addWidget(self.temp, 2, 1)
        grid.addWidget(self.ok, 3, 1)
        for data in self.Rawdata:
            if data.uid == self.listWidget_data.selectedItems()[0].data(32):
                if data.Ndop is not None:
                    self.dop.setText('{:e}'.format(data.Ndop))
                if data.temp is not None:
                    self.temp.setText(str(data.temp))
                if data.doptype == 'n':
                    self.ntype.setChecked(True)
                if data.doptype == 'p':
                    self.ptype.setChecked(True)
        self.dop.textChanged[str].connect(self.checkparam)
        self.temp.textChanged[str].connect(self.checkparam)
        self.ok.clicked.connect(self.setparam)
        self.dialog.exec_()

    def checkparam(self):
        try:
            float(self.dop.text())
            float(self.temp.text())
            self.ok.setEnabled(True)
        except ValueError:
            self.ok.setEnabled(False)

    def checkcorp(self):
        self.pushButton_corpconfirm.setEnabled(False)
        self.pushButton_corpcancel.setEnabled(False)
        try:
            if float(self.lineEdit_croplow.text()) >= float(self.lineEdit_crophigh.text()):
                self.pushButton_corp.setEnabled(False)
            elif len(self.listWidget_data.selectedItems()) > 0:
                self.pushButton_corp.setEnabled(True)
        except:
            self.pushButton_corp.setEnabled(False)

    def setparam(self):
        for data in self.Rawdata:
            if data.uid == self.listWidget_data.selectedItems()[0].data(32):
                data.Ndop = float(self.dop.text())
                data.temp = float(self.temp.text())
                if self.ntype.isChecked():
                    data.doptype = 'n'
                if self.ptype.isChecked():
                    data.doptype = 'p'
        self.updateDataList()
        self.dialog.close()

    def checkselection(self):
        if len(self.listWidget_data.selectedItems()) > 0:
            self.pushButton_del.setEnabled(True)
            self.pushButton_corp.setEnabled(True)
            self.pushButton_data2plot.setEnabled(True)
            self.pushButton_expdata.setEnabled(True)
            i = 0
            for item in self.listWidget_data.selectedItems():
                for data in self.Rawdata:
                    if data.uid == item.data(32):
                        if data.checkset() is True:
                            i += 1
            if i == 0:
                self.pushButton_data2fit.setEnabled(False)
                self.pushButton_intcorrect.setEnabled(False)
            else:
                self.pushButton_data2fit.setEnabled(True)
                self.pushButton_intcorrect.setEnabled(True)
            if len(self.listWidget_data.selectedItems()) == 1:
                self.pushButton_set.setEnabled(True)
            else:
                self.pushButton_set.setEnabled(False)
            if len(self.listWidget_data.selectedItems()) == 2:
                self.pushButton_sub2data.setEnabled(True)
                self.pushButton_mergetwo.setEnabled(True)
            else:
                self.pushButton_sub2data.setEnabled(False)
                self.pushButton_mergetwo.setEnabled(False)
        else:
            self.pushButton_sub2data.setEnabled(False)
            self.pushButton_mergetwo.setEnabled(False)
            self.pushButton_intcorrect.setEnabled(False)
            self.pushButton_corp.setEnabled(False)
            self.pushButton_expdata.setEnabled(False)
            self.pushButton_del.setEnabled(False)
            self.pushButton_data2plot.setEnabled(False)
            self.pushButton_set.setEnabled(False)
            self.pushButton_data2fit.setEnabled(False)

        if len(self.listWidget_analysis.selectedItems()) > 0:
            self.pushButton_delanalysis.setEnabled(True)
        else:
            self.pushButton_delanalysis.setEnabled(False)

        if len(self.listWidget_plotlist.selectedItems()) > 0:
            self.pushButton_delplot.setEnabled(True)
        else:
            self.pushButton_delplot.setEnabled(False)

        if self.listWidget_analysis.count() > 0:
            self.pushButton_Analysis.setEnabled(True)
        else:
            self.pushButton_Analysis.setEnabled(False)
        self.checkcorp()

    def add2analysis(self):
        for item in self.listWidget_data.selectedItems():
            for data in self.Rawdata:
                if data.uid == item.data(32):
                    if data.checkset() == True:
                        dataitem = QListWidgetItem(
                            parent=self.listWidget_analysis)
                        dataitem.setText(data.name)
                        dataitem.setData(32, data.uid)
                        self.listWidget_analysis.addItem(dataitem)

    def delAnalysis(self):
        for item in self.listWidget_analysis.selectedItems():
            self.listWidget_analysis.takeItem(
                self.listWidget_analysis.row(item))

    def export(self):
        pass

    def croppreview(self):
        self.ax1.clear()
        self.ax1.grid()
        self.ax1.set_xlabel(r'Excess carrier density $[cm^{-3}]$')
        for item in self.listWidget_data.selectedItems():
            for data in self.Rawdata:
                if data.uid == item.data(32):
                    index = data.nxc >= float(self.lineEdit_croplow.text())
                    index *= data.nxc <= float(self.lineEdit_crophigh.text())
                    tau = data.tau[index]
                    nxc = data.nxc[index]
                    if self.comboBox_plotinverse.currentIndex() == 0:
                        self.ax1.set_ylabel('Lifetime [s]')
                        self.ax1.plot(data.nxc, data.tau, '.', label=data.name)
                        self.ax1.plot(nxc, tau, 'o', label='crop_' + data.name)
                    elif self.comboBox_plotinverse.currentIndex() == 1:
                        self.ax1.set_ylabel('Inverse Lifetime [s-1]')
                        self.ax1.plot(data.nxc, 1. / data.tau,
                                      '.', label=data.name)
                        self.ax1.plot(nxc, 1. / tau, 'o',
                                      label='crop_' + data.name)
        self.ax1.semilogx()
        self.ax1.legend(loc=0)
        self.canvas.draw()
        self.figure.tight_layout()
        self.pushButton_corpconfirm.setEnabled(True)
        self.pushButton_corpcancel.setEnabled(True)
        self.comboBox_plotwhat.setEnabled(False)
        self.comboBox_plotinverse.setEnabled(False)
        self.listWidget_plotlist.setEnabled(False)
        self.pushButton_delplot.setEnabled(False)
        self.pushButton_intcorrect.setEnabled(False)
        self.pushButton_sub2data.setEnabled(False)
        self.pushButton_mergetwo.setEnabled(False)
        self.groupBox.setEnabled(False)
        self.groupBox_2.setEnabled(False)

    def crop(self):
        for item in self.listWidget_data.selectedItems():
            crop = rawdata()
            for data in self.Rawdata:
                if data.uid == item.data(32):
                    data.plotopt = False
                    index = data.nxc >= float(self.lineEdit_croplow.text())
                    index *= data.nxc <= float(self.lineEdit_crophigh.text())
                    tau = data.tau[index]
                    nxc = data.nxc[index]
                    crop.name = 'crop_' + data.name
                    crop.Ndop = data.Ndop
                    crop.temp = data.temp
                    crop.doptype = data.doptype
                    crop.nxc = nxc
                    crop.tau = tau
                    crop.PCPL = data.PCPL
                    print(type(crop))
                    print(type(self.Rawdata))
            self.Rawdata.append(crop)

        self.updateuid()
        self.updateDataList()
        self.updateplotlist()
        self.cropcancel()

    def cropcancel(self):
        self.updatedataplot()
        self.pushButton_corpconfirm.setEnabled(False)
        self.pushButton_corpcancel.setEnabled(False)
        self.comboBox_plotwhat.setEnabled(True)
        self.comboBox_plotinverse.setEnabled(True)
        self.listWidget_plotlist.setEnabled(True)
        self.groupBox.setEnabled(True)
        self.groupBox_2.setEnabled(True)
        self.checkselection()

    def correctintrinsic(self):
        pass

    def sub2(self):
        pass

    def merge2(self):
        pass

    def StartAnalysis(self):
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    LSana = LSana()
    LSana.show()
    sys.exit(app.exec_())
