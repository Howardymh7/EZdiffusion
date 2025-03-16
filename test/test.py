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
b=[]
bsq=[]
Nvalues = [10,40,4000]
for i in range(len(Nvalues)):
    N = Nvalues[i]
    b_int,bsq_int = ez.recover_ez_parameters(a,v,t,N)
    b.append(b_int)
    bsq.append(bsq_int)


class TestEZDiffusionModel(unittest.TestCase):
    def test_squared_error_1(self):
        self.assertLess(bsq[1], bsq[0], "Squared error of N=40 is not smaller than N=10")

    def test_squared_error_2(self):
        self.assertLess(bsq[2], bsq[1], "Squared error of N=4000 is not smaller than N=40")

    def test_bias_at_N10(self):
        self.assertAlmostEqual(b[0], 0, places=0, msg="bias then N=10 is not almost equal to 0")

    def test_bias_at_N40(self):
        self.assertAlmostEqual(b[1], 0, places=0, msg="bias then N=40 is not almost equal to 0")

    def test_bias_at_N4000(self):
        self.assertAlmostEqual(b[2], 0, places=0, msg="bias then N=4000 is not almost equal to 0")


# Run the tests
if __name__ == "__main__":
    unittest.main()