from .context import src

import unittest
from src.bvp_methods import *

class TestNumericalMethods(unittest.TestCase):
    
    def test_square(self):
        self.assertEqual(square(2), 4)
        self.assertEqual(square(0), 0)
        self.assertEqual(square(-2), 4)
    
    def test_factorial(self):
        self.assertEqual(factorial(5), 120)
        self.assertEqual(factorial(1), 1)
        self.assertEqual(factorial(0), 1)
    
if __name__ == '__main__':
    unittest.main()
