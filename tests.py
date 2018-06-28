from run import run
from oct2py import octave
import os
import unittest
import math

# Tests for comparing LIBOL (Python) and LIBOL (MATLAB)
class LIBOLtests(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        
        self.bc_test_dir   = './data/test/bc'
        self.bc_files      = os.listdir(self.bc_test_dir)
        #self.algorithms = ['Perceptron','Kernel_Perceptron', 'PA','PA1','PA2','OGD','Kernel_OGD','SOP','CW','SCW','SCW2','AROW','NAROW'] 
        
        # Algorithms in both LIBOL & LIBOL_py
        self.common_algorithms = ['Perceptron','PA','PA1','PA2','OGD','SOP','CW','SCW','SCW2','AROW','NAROW'] 
        
        # Compile octave modules
        print("\nCompiling modules for Octave \n")
        octave.run('./LIBOL-0.3/make.m')
        print()

    # Make sure binary classification Algorithms have the same performance, and equal or better execution times
    # BC algorithms = ['Perceptron','Kernel_Perceptron', 'PA','PA1','PA2','OGD','Kernel_OGD','SOP','CW','SCW','SCW2','AROW','NAROW'] 
    def test_bc(self):
        
        # Test all common algorithms
        for algorithm in self.common_algorithms:
            print("Testing ", algorithm ," (bc)")
            
            # Test all files in directory
            for filename in self.bc_files:
                
                
                path = self.bc_test_dir + '/' + filename
                path_octave = '/test/bc/' + filename
                
                # Octave execution
                mean_err_oct, std_error_oct, mean_nSV_oct, std_nSV_oct, mean_time_oct, std_time_oct = octave.feval('./LIBOL-0.3/demo.m', 'bc',algorithm,path_octave,'libsvm', nout = 6)
                
                # Python execution
                result_python = run('bc', algorithm, path, 'libsvm', print_trials = False, print_results = False)
                mean_err_py  = result_python[0]
                mean_nSV_py  = result_python[1]
                mean_time_py = result_python[2]
                
                # Test for similar performance
                error_diff = abs(mean_err_oct - mean_err_py) 
                self.assertLessEqual(error_diff, 6*std_error_oct)
                
                # Test for better execution time
                self.assertLessEqual(mean_time_py, mean_time_oct)

    #def test_Kernel_Perceptron(self):
    #    pass
        #print("a")

    # Test performance against LIBOL for matlab
    #def test_bc_performance(self):
        #libol_py_results = compare('bc','./data/bc/breast_cancer_scale.txt','libsvm', showPlot = False)
        #octave.run("cos(pi / 3)")
        #r = octave.feval("cos",3.1415926538/3)
        #print(libol_py_results)
        #self.assertEqual(1, 1)
        
    def tearDown(self):
        pass
        # Remove LIBOL release for MATLAB
        #if (os.path.isdir('./LIBOL-0.3')):
        #    os.system('rm -r ./LIBOL-0.3')
    
if __name__ == '__main__':
    unittest.main()
