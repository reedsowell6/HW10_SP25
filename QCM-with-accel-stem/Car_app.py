# Car_app.py
#region imports
from Car_GUI import Ui_Form
import sys
from PyQt5 import QtCore as qtc, QtWidgets as qtw
from quarter_car_model import CarController
# matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
#endregion

class MainWindow(qtw.QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # pack widget refs
        inputs = (
            self.le_m1, self.le_v, self.le_k1, self.le_c1,
            self.le_m2, self.le_k2, self.le_ang, self.le_tmax,
            self.chk_IncludeAccel
        )
        displays = (
            self.gv_Schematic, self.chk_LogX, self.chk_LogY,
            self.chk_LogAccel, self.chk_ShowAccel,
            self.lbl_MaxMinInfo, self.layout_horizontal_main
        )
        self.controller = CarController(inputs, displays)

        # connect signals
        self.btn_calculate.clicked.connect(self.controller.calculate)
        self.pb_Optimize.clicked.connect(self.controller.optimise)
        for chk in (self.chk_LogX, self.chk_LogY, self.chk_LogAccel, self.chk_ShowAccel):
            chk.stateChanged.connect(self.controller.plot)

        self.show()

if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    win = MainWindow()
    win.setWindowTitle('Quarter Car Model')
    sys.exit(app.exec_())
