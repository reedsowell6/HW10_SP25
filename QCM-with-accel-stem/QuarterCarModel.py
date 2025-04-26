# QuarterCarModel.py
#region imports
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import math
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
#endregion

#region class definitions
class MassBlock(qtw.QGraphicsItem):
    # ... (unchanged drawing code) ...
    pass

class Wheel(qtw.QGraphicsItem):
    # ... (unchanged drawing code) ...
    pass

class CarModel():
    def __init__(self):
        self.results = None
        self.tmax = 3.0
        self.t = np.linspace(0, self.tmax, 200)
        self.tramp = 1.0
        self.yangdeg = 45.0
        self.angrad = self.yangdeg * math.pi / 180.0
        self.ymag = 6.0 / (12 * 3.3)

        # default parameters (will be overwritten)
        self.m1 = 450.0
        self.m2 = 20.0
        self.c1 = 4500.0
        self.k1 = 15000.0
        self.k2 = 90000.0
        self.v  = 120.0

        # static deflection bounds and accel limits
        self.mink1 = 0.0
        self.maxk1 = 0.0
        self.mink2 = 0.0
        self.maxk2 = 0.0
        self.accelLim = 2.0
        self.accelMax = 0.0
        self.SSE = 0.0

class CarView():
    def __init__(self, args):
        # ... (unchanged view code) ...
        pass

    def updateView(self, model=None):
        # ... (unchanged updateView code) ...
        pass

    def buildScene(self):
        # ... (unchanged buildScene code) ...
        pass

    def doPlot(self, model=None):
        # ... (unchanged plotting code) ...
        pass

class CarController():
    def __init__(self, args):
        self.input_widgets, self.display_widgets = args
        (self.le_m1, self.le_v, self.le_k1, self.le_c1,
         self.le_m2, self.le_k2, self.le_ang,
         self.le_tmax, self.chk_IncludeAccel) = self.input_widgets

        (self.gv_Schematic, self.chk_LogX, self.chk_LogY,
         self.chk_LogAccel, self.chk_ShowAccel,
         self.lbl_MaxMinInfo, self.layout_horizontal_main) = self.display_widgets

        self.model = CarModel()
        self.view  = CarView(args)

    def ode_system(self, X, t):
        if t < self.model.tramp:
            y = self.model.ymag * (t / self.model.tramp)
        else:
            y = self.model.ymag

        x1, x1dot, x2, x2dot = X

        # suspension and tire forces
        f_susp = self.model.k1*(x1 - x2) + self.model.c1*(x1dot - x2dot)
        f_tire = self.model.k2*(x2 - y)

        # equations of motion
        x1ddot = -f_susp / self.model.m1
        x2ddot = ( f_susp - f_tire ) / self.model.m2

        return [x1dot, x1ddot, x2dot, x2ddot]

    def calculate(self, doCalc=True):
        # read inputs
        self.model.m1 = float(self.le_m1.text())
        self.model.m2 = float(self.le_m2.text())
        self.model.c1 = float(self.le_c1.text())
        self.model.k1 = float(self.le_k1.text())
        self.model.k2 = float(self.le_k2.text())
        self.model.v  = float(self.le_v.text())

        # compute static deflection bounds
        inch = 0.0254
        δ1_min = 3.0*inch
        δ1_max = 6.0*inch
        self.model.mink1 = self.model.m1*9.81/δ1_max
        self.model.maxk1 = self.model.m1*9.81/δ1_min
        δ2_min = 0.75*inch
        δ2_max = 1.5*inch
        self.model.mink2 = self.model.m2*9.81/δ2_max
        self.model.maxk2 = self.model.m2*9.81/δ2_min

        # other settings
        self.model.accelLim = 2.0
        self.model.ymag = 6.0/(12.0*3.3)
        self.model.yangdeg = float(self.le_ang.text())
        self.model.tmax   = float(self.le_tmax.text())

        if doCalc:
            self.doCalc()

        # compute SSE for display
        self.SSE((self.model.k1, self.model.c1, self.model.k2), optimizing=False)
        self.view.updateView(self.model)

    def doCalc(self, doPlot=True, doAccel=True):
        v = 1000*self.model.v/3600.0
        self.model.angrad = math.radians(self.model.yangdeg)
        self.model.tramp = self.model.ymag / (math.sin(self.model.angrad)*v)

        self.model.t = np.linspace(0, self.model.tmax, 2000)
        ic = [0,0,0,0]
        self.model.results = odeint(self.ode_system, ic, self.model.t)

        if doAccel:
            self.calcAccel()
        if doPlot:
            self.doPlot()

    def calcAccel(self):
        N = len(self.model.t)
        vel = self.model.results[:,1]
        accel = np.zeros(N)
        for i in range(N):
            if i == N-1:
                h = self.model.t[i] - self.model.t[i-1]
                accel[i] = (vel[i]-vel[i-1])/(9.81*h)
            else:
                h = self.model.t[i+1] - self.model.t[i]
                accel[i] = (vel[i+1]-vel[i])/(9.81*h)
        self.model.accel    = accel
        self.model.accelMax = accel.max()
        return True

    def SSE(self, vals, optimizing=True):
        k1, c1, k2 = vals
        self.model.k1 = k1
        self.model.c1 = c1
        self.model.k2 = k2
        self.doCalc(doPlot=False)

        SSE = 0.0
        for i, yi in enumerate(self.model.results[:,0]):
            t = self.model.t[i]
            if t < self.model.tramp:
                y_tgt = self.model.ymag*(t/self.model.tramp)
            else:
                y_tgt = self.model.ymag
            SSE += (yi - y_tgt)**2

        if optimizing:
            if k1 < self.model.mink1 or k1 > self.model.maxk1:
                SSE += 100
            if c1 < 10:
                SSE += 100
            if k2 < self.model.mink2 or k2 > self.model.maxk2:
                SSE += 100
            if self.chk_IncludeAccel.isChecked() and self.model.accelMax > self.model.accelLim:
                SSE += (self.model.accelMax - self.model.accelLim)**2

        self.model.SSE = SSE
        return SSE

    def OptimizeSuspension(self):
        # ensure current inputs are loaded
        self.calculate(doCalc=False)

        # initial guess
        x0 = np.array([self.model.k1, self.model.c1, self.model.k2])

        # run optimizer
        res = minimize(self.SSE, x0, method='Nelder-Mead', options={'maxiter':500, 'disp':True})

        # apply optimal values
        self.SSE(res.x, optimizing=False)
        self.view.updateView(self.model)

    def doPlot(self):
        self.view.doPlot(self.model)

#endregion