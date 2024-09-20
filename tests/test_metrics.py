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
        expected_spd = {'target1': {'sensitive1': -0.5}}
    
        spd_result = self.df.statistical_parity_difference()
        assert spd_result == expected_spd, f"Expected {expected_spd}, but got {spd_result}"
    
        self.df.targets = ['target1', 'target2']
        self.df.sensitive = ['sensitive1', 'sensitive2', 'sensitive3']
    
        expected_spd = {
            'target1': {
                'sensitive1': -0.500,  
                'sensitive2': 0.500,   
                'sensitive3': 0.267
            },
            'target2': {
                'sensitive1': 0.250,  
                'sensitive2': -0.250, 
                'sensitive3': -0.067
            }
        }   
    
        spd_result = self.df.statistical_parity_difference()
        for target in expected_spd:
            for sensitive in expected_spd[target]:
                self.assertAlmostEqual(spd_result[target][sensitive], expected_spd[target][sensitive], places=3, 
                                        msg=f"Expected {expected_spd[target][sensitive]} for {target}-{sensitive}, but got {spd_result[target][sensitive]}")
        
    def testDisparateImpact(self):
        self.df.targets = ['target1']
        self.df.sensitive = ['sensitive1']
        expected_di = {'target1': {'sensitive1': 3.0}}
        
        di_result = self.df.disparate_impact()
        assert di_result == expected_di, f"Expected {expected_di}, but got {di_result}"
        
        self.df.targets = ['target1', 'target2']
        self.df.sensitive = ['sensitive1', 'sensitive2', 'sensitive3']
        
        expected_di = {
            'target1': {
                'sensitive1': 3.000, 
                'sensitive2': 0.333,
                'sensitive3': 0.556
            },
            'target2': {
                'sensitive1': 0.667,
                'sensitive2': 1.500,
                'sensitive3': 1.111
            }
        }
        
        di_result = self.df.disparate_impact()
        for target in expected_di:
            for sensitive in expected_di[target]:
                self.assertAlmostEqual(di_result[target][sensitive], expected_di[target][sensitive], places=3, 
                                       msg=f"Expected {expected_di[target][sensitive]} for {target}-{sensitive}, but got {di_result[target][sensitive]}")
        
        

    def tearDown(self):
        del self.df

if __name__ == '__main__':
    unittest.main()

