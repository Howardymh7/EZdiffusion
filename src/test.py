import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ez_diffusion_model import EZDiffusionModel
ez = EZDiffusionModel()

a = np.random.uniform(0.5, 2)
v = np.random.uniform(0.5, 2)
t = np.random.uniform(0.1, 0.5)
N1 = 10
N2 = 40
N3 = 4000
ez.recover_ez_parameters(a,v,t,N)

class TestEZDiffusionModel(unittest.TestCase):
    def setUp(self):
        self.ez = EZDiffusionModel()