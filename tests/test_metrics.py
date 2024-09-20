import unittest
import fairlib as fl
import fairlib.metrics
import pandas as pd

class TestMetrics(unittest.TestCase):
    
    def setUp(self):
        '''        
           'target1': [1, 0, 1, 1, 0, 1, 0, 0],
        'sensitive1': [1, 1, 0, 0, 1, 0, 0, 1],
        
        Example: 
        Privileged group (sensitive == 1): target = [1, 0, 0, 0]
	        Average = (1 + 0 + 0) / 4 = 0.25
	    Unprivileged group (sensitive == 0): target = [1, 1, 1, 0]
	        Mean = (1 + 1 + 0) / 4 = 0.75
         
        SPD = 0.25 - 0.75 = -0.5
        DI = 0.75/0.25 = 3.0
        '''
        self.df = fl.DataFrame({
               'target1': [1, 0, 1, 1, 0, 1, 0, 0],
               'target2': [0, 1, 0, 1, 1, 0, 1, 1],
            'sensitive1': [1, 1, 0, 0, 1, 0, 0, 1],
            'sensitive2': [0, 0, 1, 1, 0, 1, 1, 0],
            'sensitive3': [1, 0, 1, 1, 1, 0, 0, 1],
        })

  
    def testStatisticalParityDifference(self):
        self.df.targets = ['target1']
        self.df.sensitive = ['sensitive1']
        expected_spd = {'target1-sensitive1': -0.5}
        
        spd_result = self.df.statistical_parity_difference()
        assert spd_result == expected_spd, f"Expected {expected_spd}, but got {spd_result}"
        
        self.df.targets = ['target1', 'target2']
        self.df.sensitive = ['sensitive1', 'sensitive2', 'sensitive3']
        
        expected_spd = {
            'target1-sensitive1': -0.500,  
            'target1-sensitive2': 0.500,   
            'target1-sensitive3': 0.267, 
            'target2-sensitive1': 0.250,  
            'target2-sensitive2': -0.250, 
            'target2-sensitive3': -0.067
        }   
        
        spd_result = self.df.statistical_parity_difference()
        for key in expected_spd:
            self.assertAlmostEqual(spd_result[key], expected_spd[key], places=3, 
                                   msg=f"Expected {expected_spd[key]} for {key}, but got {spd_result[key]}")
        
    def testDisparateImpact(self):
        self.df.targets = ['target1']
        self.df.sensitive = ['sensitive1']
        expected_di = {'target1-sensitive1': 3.0}
        
        di_result = self.df.disparate_impact()
        assert di_result == expected_di, f"Expected {expected_di}, but got {di_result}"
        
        self.df.targets = ['target1', 'target2']
        self.df.sensitive = ['sensitive1', 'sensitive2', 'sensitive3']
        
        expected_di = {
            'target1-sensitive1': 3.000, 
            'target1-sensitive2': 0.333,
            'target1-sensitive3': 0.556,
            'target2-sensitive1': 0.667,
            'target2-sensitive2': 1.500,
            'target2-sensitive3': 1.111,
        }
        
        di_result = self.df.disparate_impact()
        for key in expected_di:
            self.assertAlmostEqual(di_result[key], expected_di[key], places=3, 
                                   msg=f"Expected {expected_di[key]} for {key}, but got {di_result[key]}")
        
        

    def tearDown(self):
        del self.df

if __name__ == '__main__':
    unittest.main()

