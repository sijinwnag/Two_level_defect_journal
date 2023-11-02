import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from uiMain import Ui_LSanalyzer
from LifeSimu import LSimu
from IDLSanalyzer import LSana


class LSanalyzerMain(QMainWindow, Ui_LSanalyzer):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.MainlifesimuPushB.clicked.connect(self.OpenSimu)
        self.MainIDLSPushB.clicked.connect(self.OpenIDLS)

        self.LSimu = None
        self.IDLSanalyzer = None

    def OpenSimu(self):
        if self.LSimu == None:
            self.LSimu = LSimu(self)
        self.LSimu.show()

    def OpenIDLS(self):
        if self.IDLSanalyzer == None:
            self.IDLSanalyzer = LSana(self)
        self.IDLSanalyzer.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    LSanalyzerMain = LSanalyzerMain()
    LSanalyzerMain.show()
    sys.exit(app.exec_())
