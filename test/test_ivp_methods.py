from context import src
import unittest
from src.ivp_methods import Euler, IEuler, Midpoint, RK4, RK45
import numpy as np

class TestEuler(unittest.TestCase):

    def test_step(self):
        # Test if the method returns the correct solution
        f = lambda t, y: y
        y0 = 1
        t0 = 0
        tf = 1
        dt = 0.1
        ode = Euler(f, y0, t0, tf, dt)
        ode.step(0)
        self.assertAlmostEqual(ode.y[1][0], 1.1, places=2)

    def test_step_size_control(self):
        # Test if the method returns the correct step size
        f = lambda t, y: y
        y0 = np.array([1])
        t0 = 0
        tf = 1
        dt = 0.1
        tol = 0.01
        ode = Euler(f, y0, t0, tf, dt)
        dt_new = ode._step_size_control(y0, t0, f, dt, tol)
        self.assertAlmostEqual(dt_new, 0.0316, places=2)

    def test_euler_step(self):
        # Test if the method returns the correct solution
        f = lambda t, y: y
        y = 1
        t = 0
        dt = 0.1
        ode = Euler(f, y, t, t+dt, dt)
        y_new = ode._euler_step(y, t, f, dt)
        self.assertAlmostEqual(y_new, 1.1, places=2)

class TestIEuler(unittest.TestCase):
    
    def test_step(self):
        # Test if the method returns the correct solution
        f = lambda t, y: y
        y0 = 1
        t0 = 0
        tf = 1
        dt = 0.1
        ode = IEuler(f, y0, t0, tf, dt)
        ode.step(0)
        self.assertAlmostEqual(ode.y[1][0], 1.1, places=2)

    def test_step_size_control(self):
        # Test if the method returns the correct step size
        f = lambda t, y: y
        y0 = np.array([1])
        t0 = 0
        tf = 1
        dt = 0.1
        tol = 0.01
        ode = IEuler(f, y0, t0, tf, dt)
        dt_new = ode._step_size_control(y0, t0, f, dt, tol)
        self.assertAlmostEqual(dt_new, 0.0316, places=2)

    def test_ieuler_step(self):
        # Test if the method returns the correct solution
        f = lambda t, y: y
        y = 1
        t = 0
        dt = 0.1
        ode = IEuler(f, y, t, t+dt, dt)
        y_new = ode._ieuler_step(y, t, f, dt)
        self.assertAlmostEqual(y_new, 1.1, places=2)

class TestMidpoint(unittest.TestCase):
        
    def test_step(self):
        # Test if the method returns the correct solution
        f = lambda t, y: y
        y0 = 1
        t0 = 0
        tf = 1
        dt = 0.1
        ode = Midpoint(f, y0, t0, tf, dt)
        ode.step(0)
        self.assertAlmostEqual(ode.y[1][0], 1.1, places=2)

    def test_step_size_control(self):
        # Test if the method returns the correct step size
        f = lambda t, y: y
        y0 = np.array([1])
        t0 = 0
        tf = 1
        dt = 0.1
        tol = 0.01
        ode = Midpoint(f, y0, t0, tf, dt)
        dt_new = ode._step_size_control(y0, t0, f, dt, tol)
        self.assertAlmostEqual(dt_new, 0.0316, places=2)

    def test_midpoint_step(self):
        # Test if the method returns the correct solution
        f = lambda t, y: y
        y = 1
        t = 0
        dt = 0.1
        ode = Midpoint(f, y, t, t+dt, dt)
        y_new = ode._midpoint_step(y, t, f, dt)
        self.assertAlmostEqual(y_new, 1.1, places=2)

class TestRK4(unittest.TestCase):
            
    def test_step(self):
        # Test if the method returns the correct solution
        f = lambda t, y: y
        y0 = 1
        t0 = 0
        tf = 1
        dt = 0.1
        ode = RK4(f, y0, t0, tf, dt, tol=1e-3)
        ode.step(0)
        self.assertAlmostEqual(ode.y[1][0], 1.1, places=1)

    def test_step_size_control(self):
        # Test if the method returns the correct step size
        f = lambda t, y: y
        y0 = np.array([1])
        t0 = 0
        tf = 10
        dt = 0.1
        tol = 1e-5
        ode = RK4(f, y0, t0, tf, dt)
        dt_new = ode._step_size_control(y0, t0, f, dt, tol)
        self.assertAlmostEqual(dt_new, 0.0316, places=2)

    def test_rk4_step(self):
        # Test if the method returns the correct solution
        f = lambda t, y: y
        y = 1
        t = 0
        dt = 0.1
        ode = RK4(f, y, t, t+dt, dt)
        y_new = ode._rk4_step(y, t, f, dt)
        self.assertAlmostEqual(y_new, 1.1, places=1)

class TestRK45(unittest.TestCase):
                
    def test_step(self):
        # Test if the method returns the correct solution
        f = lambda t, y: y
        y0 = 1
        t0 = 0
        tf = 1
        dt = 0.1
        ode = RK45(f, y0, t0, tf, dt, tol=1e-1)
        ode.step(0)
        self.assertAlmostEqual(ode.y[1][0], 1.1, places=1)

    def test_step_size_control(self):
        # Test if the method returns the correct step size
        f = lambda t, y: y
        y0 = np.array([1])
        t0 = 0
        tf = 100
        dt = 0.1
        tol = 1e-4
        ode = RK45(f, y0, t0, tf, dt)
        dt_new = ode._step_size_control(y0, t0, f, dt, tol)
        self.assertAlmostEqual(dt_new, 0.0316, places=2)

    def test_rk45_step(self):
        # Test if the method returns the correct solution
        f = lambda t, y: y
        y = 1
        t = 0
        dt = 0.1
        ode = RK45(f, y, t, t+dt, dt, tol=1e-6)
        y_new = ode._rk45_step(y, t, f, dt)
        self.assertAlmostEqual(y_new, 1.1, places=1)

if __name__ == '__main__':
    unittest.main()

