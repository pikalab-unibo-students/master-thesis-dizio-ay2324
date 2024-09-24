import unittest
import fairlib as fl
import fairlib.pre_processing
import pandas.testing as pd_testing

class TestPreProcessing(unittest.TestCase):
    
    def setUp(self):
        self.df = fl.DataFrame({
               'target1': [0,1,0,1,0,1],
               'target2': [1,0,1,0,1,0],
            'sensitive1': [0,1,0,1,0,0]
        })

  
    def testSingleReweighin(self):
        self.df.targets = {'target1'}
        self.df.sensitive = {'sensitive1'}
        transformed_df = self.df.reweighing()
        expected_df = fl.DataFrame({
               'target1': [0,1,0,1,0,1],
               'target2': [1,0,1,0,1,0],
            'sensitive1': [0,1,0,1,0,0],
            'weights': [0.666667, 0.500000, 0.666667, 0.500000, 0.666667, 2.000000]
        })
        pd_testing.assert_frame_equal(transformed_df, expected_df)
        
        self.df.targets = {'target1', 'target2'}
        self.df.sensitive = {'sensitive1'}
        
        with self.assertRaises(ValueError):
            self.df.reweighing()
       
    def tearDown(self):
        del self.df

if __name__ == '__main__':
    unittest.main()

