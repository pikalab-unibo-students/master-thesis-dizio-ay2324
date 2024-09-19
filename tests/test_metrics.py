import unittest
import fairlib as fl
import fairlib.metrics
import pandas as pd

class TestMetrics(unittest.TestCase):
    
    def setUp(self):
        '''
        Create a DataFrame with 1 sensitive attributes and 1 target attribute:
        
           'target': [1, 0, 1, 1, 0, 1, 0, 0],
        'sensitive': [1, 1, 0, 0, 1, 0, 0, 1],
        
        Example: 
        Privileged group (sensitive == 1): target = [1, 0, 0, 0]
	        Average = (1 + 0 + 0) / 4 = 0.25
	    Unprivileged group (sensitive == 0): target = [1, 1, 1, 0]
	        Mean = (1 + 1 + 0) / 4 = 0.75
         
        SPD = 0.25 - 0.75 = -0.5
        DI = 0.75/0.25 = 3.0
        '''
        self.df = fl.DataFrame({
               'target': [1, 0, 1, 1, 0, 1, 0, 0],
            'sensitive': [1, 1, 0, 0, 1, 0, 0, 1],
        })
        self.df.targets = ['target']
        self.df.sensitive = ['sensitive']
  
    def testStatisticalParityDifference(self):
        expected_spd = {'target-sensitive': -0.5}
        
        spd_result = self.df.statistical_parity_difference()
        assert spd_result == expected_spd, f"Expected {expected_spd}, but got {spd_result}"
        
    def testDisparateImpact(self):
        expected_di = {'sensitive': 3.0}
        
        di_result = self.df.disparate_impact()
        assert di_result == expected_di, f"Expected {expected_di}, but got {di_result}"

    def tearDown(self):
        del self.df

if __name__ == '__main__':
    unittest.main()

