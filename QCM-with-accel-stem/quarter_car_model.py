# quarter_car_model.py
#region imports
import numpy as np
import math
from scipy.integrate import odeint
from scipy.optimize import minimize
from PyQt5 import QtWidgets as qtw, QtGui as qtg, QtCore as qtc
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
#endregion

class MassBlock(qtw.QGraphicsItem):
    def __init__(self, x, y, w=30, h=10, pen=None, brush=None):
        super().__init__()
        self.x, self.y = x, y
        self.pen, self.brush = pen, brush
        self.rect = qtc.QRectF(-w/2, -h/2, w, h)
    def boundingRect(self): return self.rect
    def paint(self, p, o, w):
        if self.pen: p.setPen(self.pen)
        if self.brush: p.setBrush(self.brush)
        p.drawRect(self.rect)
        self.setTransform(qtg.QTransform().translate(self.x, self.y))

class Wheel(qtw.QGraphicsItem):
    def __init__(self, x, y, r=20, pen=None, wheelBrush=None, massBrush=None):
        super().__init__()
        self.x, self.y = x, y
        self.pen, self.brush = pen, wheelBrush
        self.rect = qtc.QRectF(-r, -r, 2*r, 2*r)
        self.massBlock = MassBlock(x, y, w=r*1.5, h=r/3, pen=pen, brush=massBrush)
    def boundingRect(self): return self.rect
    def paint(self, p, o, w):
        if self.pen: p.setPen(self.pen)
        if self.brush: p.setBrush(self.brush)
        p.drawEllipse(self.rect)
        self.setTransform(qtg.QTransform().translate(self.x, self.y))
    def addToScene(self, scene):
        scene.addItem(self)
        scene.addItem(self.massBlock)

class SpringItem(qtw.QGraphicsItem):
    def __init__(self, p1, p2, coils=8, amplitude=5, pen=None):
        super().__init__()
        self.p1 = qtc.QPointF(*p1)
        self.p2 = qtc.QPointF(*p2)
        self.coils = coils
        self.amp = amplitude
        self.pen = pen
        self.path = self.build_path()
    def build_path(self):
        path = qtg.QPainterPath(self.p1)
        vec = self.p2 - self.p1
        length = math.hypot(vec.x(), vec.y())
        dir = vec / length
        perp = qtc.QPointF(-dir.y(), dir.x())
        step = length / (self.coils * 2)
        pt = qtc.QPointF(self.p1)
        for i in range(self.coils * 2):
            pt += dir * step
            off = perp * (self.amp if i % 2 else -self.amp)
            path.lineTo(pt + off)
        path.lineTo(self.p2)
        return path
    def boundingRect(self): return self.path.boundingRect()
    def paint(self, p, o, w):
        if self.pen: p.setPen(self.pen)
        p.drawPath(self.path)

class DashpotItem(qtw.QGraphicsItem):
    def __init__(self, p1, p2, width=10, pen=None, brush=None):
        super().__init__()
        self.p1 = qtc.QPointF(*p1)
        self.p2 = qtc.QPointF(*p2)
        self.width = width
        self.pen = pen
        self.brush = brush
        self.path, self.rect_body = self.build_path()
    def build_path(self):
        path = qtg.QPainterPath()
        path.moveTo(self.p1)
        path.lineTo(self.p2)
        mid = (self.p1 + self.p2) / 2
        size = self.width
        rect = qtc.QRectF(mid.x()-size/2, mid.y()-size/4, size, size/2)
        return path, rect
    def boundingRect(self):
        br = self.path.boundingRect()
        return br.united(self.rect_body)
    def paint(self, p, o, w):
        if self.pen: p.setPen(self.pen)
        if self.brush: p.setBrush(self.brush)
        p.drawPath(self.path)
        p.drawRect(self.rect_body)

class CarModel:
    def __init__(self):
        self.m1, self.m2 = 450.0, 20.0
        self.k1, self.k2 = 15000.0, 90000.0
        self.c1 = 4500.0
        self.v = 120.0
        self.yangdeg = 45.0
        self.ymag = 6.0/(12*3.3)
        self.angrad = math.radians(self.yangdeg)
        self.tmax = 3.0
        self.t = np.linspace(0, self.tmax, 2000)
        self.compute_bounds()
        self.accelLim = 2.0
        self.results = None
        self.accel = None
        self.accelMax = 0.0
        self.SSE = np.inf
    def compute_tramp(self):
        v_ms = self.v*1000/3600
        return self.ymag/(v_ms*math.sin(self.angrad)) if v_ms>0 else np.inf
    def compute_bounds(self):
        inch = 0.0254
        d1min, d1max = 3*inch, 6*inch
        d2min, d2max = 0.75*inch, 1.5*inch
        self.mink1 = self.m1*9.81/d1max
        self.maxk1 = self.m1*9.81/d1min
        self.mink2 = self.m2*9.81/d2max
        self.maxk2 = self.m2*9.81/d2min

class CarView:
    def __init__(self, inputs, displays):
        (self.le_m1, self.le_v, self.le_k1, self.le_c1,
         self.le_m2, self.le_k2, self.le_ang, self.le_tmax,
         self.chk_IncludeAccel) = inputs
        (self.gv, self.chk_LogX, self.chk_LogY,
         self.chk_LogAccel, self.chk_ShowAccel,
         self.lbl_Info, self.layout) = displays
        self.fig = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.layout.addWidget(self.canvas)
        self.ax = self.fig.add_subplot()
        self.ax1 = self.ax.twinx()
        self.build_scene()
    def build_scene(self):
        scene = qtw.QGraphicsScene()
        scene.setSceneRect(-200, -200, 400, 400)
        self.gv.setScene(scene)
        pen = qtg.QPen(qtg.QColor('orange')); pen.setWidth(2)
        bw = qtg.QBrush(qtg.QColor.fromHsv(35,255,255,64))
        bm = qtg.QBrush(qtg.QColor(200,200,200,128))
        body_pos = (0, -70)
        wheel_pos= (0,  50)
        body = MassBlock(*body_pos, w=100, h=30, pen=pen, brush=bm)
        wheel= Wheel(*wheel_pos, r=50, pen=pen, wheelBrush=bw, massBrush=bm)
        p1 = (body_pos[0], body_pos[1]+15)
        p2 = (wheel_pos[0], wheel_pos[1]-50)
        spring = SpringItem(p1, p2, coils=6, amplitude=8, pen=pen)
        dp1 = (body_pos[0]+20, body_pos[1]+15)
        dp2 = (wheel_pos[0]+20, wheel_pos[1]-50)
        dashpot = DashpotItem(dp1, dp2, width=12, pen=pen, brush=None)
        scene.addItem(spring)
        scene.addItem(dashpot)
        scene.addItem(body)
        wheel.addToScene(scene)
    def update(self, model):
        self.le_m1.setText(f"{model.m1:.1f}")
        self.le_k1.setText(f"{model.k1:.1f}")
        self.le_c1.setText(f"{model.c1:.1f}")
        self.le_m2.setText(f"{model.m2:.1f}")
        self.le_k2.setText(f"{model.k2:.1f}")
        self.le_ang.setText(f"{model.yangdeg:.1f}")
        self.le_tmax.setText(f"{model.tmax:.1f}")
        info = (f"k1:[{model.mink1:.0f},{model.maxk1:.0f}], k2:[{model.mink2:.0f},{model.maxk2:.0f}]\n"
                f"SSE={model.SSE:.1f}, a_max={model.accelMax:.2f}g")
        self.lbl_Info.setText(info)
        self.plot(model)
    def plot(self, model):
        if model.results is None: return
        t= model.t; y1=model.results[:,0]; y2=model.results[:,2]; a=model.accel
        ax,ax1=self.ax,self.ax1; ax.clear(); ax1.clear()
        ax.set_xscale('log' if self.chk_LogX.isChecked() else 'linear')
        ax.set_yscale('log' if self.chk_LogY.isChecked() else 'linear')
        ax.plot(t,y1,'b-'); ax.plot(t,y2,'r-')
        if self.chk_ShowAccel.isChecked(): ax1.plot(t,a,'g-')
        ax.axhline(model.ymag); ax.axvline(model.tramp)
        self.canvas.draw()

class CarController:
    def __init__(self, inputs, displays):
        self.model = CarModel()
        self.view  = CarView(inputs, displays)

    def read_fields(self):
        # Read GUI input values
        m1   = float(self.view.le_m1.text())
        m2   = float(self.view.le_m2.text())
        k1   = float(self.view.le_k1.text())
        k2   = float(self.view.le_k2.text())
        c1   = float(self.view.le_c1.text())
        v    = float(self.view.le_v.text())
        ang  = float(self.view.le_ang.text())
        tmax = float(self.view.le_tmax.text())

        # Update model parameters
        self.model.m1      = m1
        self.model.m2      = m2
        self.model.k1      = k1
        self.model.k2      = k2
        self.model.c1      = c1
        self.model.v       = v
        self.model.yangdeg = ang
        self.model.tmax    = tmax
        # Recompute derived values
        self.model.angrad  = math.radians(ang)
        self.model.ymag    = 6.0 / (12 * 3.3)
        self.model.compute_bounds()
        self.model.tramp   = self.model.compute_tramp()
        self.model.t       = np.linspace(0, self.model.tmax, 2000)

    def ode(self, X, t):
        # Quarter-car ODEs: X = [x1, x1dot, x2, x2dot]
        # Road profile
        y = (self.model.ymag * (t/self.model.tramp)) if t < self.model.tramp else self.model.ymag
        x1, x1d, x2, x2d = X
        # suspension and tire forces
        fs = self.model.k1*(x1 - x2) + self.model.c1*(x1d - x2d)
        ft = self.model.k2*(x2 - y)
        # accelerations
        x1dd = -fs / self.model.m1
        x2dd = (fs - ft) / self.model.m2
        return [x1d, x1dd, x2d, x2dd]

    def calculate(self):
        # load fields and update model
        self.read_fields()
        # solve ODE
        sol = odeint(self.ode, [0,0,0,0], self.model.t)
        self.model.results = sol
        # compute acceleration
        vel = sol[:,1]
        dt = np.diff(self.model.t)
        acc = np.concatenate([ (vel[1:] - vel[:-1])/(9.81*dt), [0] ])
        self.model.accel    = acc
        self.model.accelMax = np.max(acc)
        # compute SSE
        self.model.SSE = self._compute_SSE((self.model.k1, self.model.c1, self.model.k2))
        # update view
        self.view.update(self.model)

    def _compute_SSE(self, vals):
        k1, c1, k2 = vals
        # backup old
        old = (self.model.k1, self.model.c1, self.model.k2)
        # temporarily assign new
        self.model.k1, self.model.c1, self.model.k2 = k1, c1, k2
        # solve ODE
        sol = odeint(self.ode, [0,0,0,0], self.model.t)
        y = sol[:,0]
        t = self.model.t
        y_tgt = np.where(t < self.model.tramp,
                         self.model.ymag * (t/self.model.tramp),
                         self.model.ymag)
        sse = np.sum((y - y_tgt)**2)
        # apply penalties
        if k1 < self.model.mink1 or k1 > self.model.maxk1:
            sse += 1e3
        if k2 < self.model.mink2 or k2 > self.model.maxk2:
            sse += 1e3
        if c1 < 10:
            sse += 1e3
        if self.view.chk_IncludeAccel.isChecked() and self.model.accelMax > self.model.accelLim:
            sse += (self.model.accelMax - self.model.accelLim)**2 * 1e2
        # restore old
        self.model.k1, self.model.c1, self.model.k2 = old
        return sse

    def optimise(self):
        # initial guess from model
        self.read_fields()
        x0 = [self.model.k1, self.model.c1, self.model.k2]
        res = minimize(self._compute_SSE, x0, method='Nelder-Mead', options={'maxiter':500, 'disp':True})
        # assign optimal
        self.model.k1, self.model.c1, self.model.k2 = res.x
        # recalc and update
        self.calculate()

    def plot(self):
        # simply forward to view
        self.view.plot(self.model)
