from run import run
from compare import compare
from oct2py import octave
import os
import unittest
import math

# Tests for comparing LIBOL (Python) and LIBOL (MATLAB)
class LIBOLtests(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        
        self.bc_test_dir   = './data/test/bc'
        self.mc_test_dir   = './data/test/mc'
        self.bc_files      = os.listdir(self.bc_test_dir)
        self.mc_files      = os.listdir(self.mc_test_dir)
        
        # Algorithms in both LIBOL & LIBOL_py
        self.common_algorithms_bc = ['Perceptron','PA','PA1','PA2','OGD','SOP','CW','SCW','SCW2','AROW','NAROW'] 
        self.common_algorithms_mc = ['M_AROW','M_CW','M_OGD','M_PA','M_PA1','M_PA2','M_PerceptronM','M_PerceptronS','M_PerceptronU','M_SCW1','M_SCW2'] 
        self.kernel_algorithms    = ['Gaussian_Kernel_Perceptron', 'Gaussian_Kernel_OGD']
        self.OGD_algorithms       = ['OGD', 'Gaussian_Kernel_OGD']
        
        # Compile octave modules
        print("\nCompiling modules for Octave \n")
        octave.run('./LIBOL-0.3/make.m')
        print()

    # Test for similar performance and execution times for binary classification algorithms with random dataset sampling
    def test_bc(self):
        
        # Test all common algorithms
        for algorithm in self.common_algorithms_bc:
            print("Testing ", algorithm ," (bc)")
            
            # Test all files in directory
            for filename in self.bc_files:
                
                
                path = self.bc_test_dir + '/' + filename
                path_octave = '/test/bc/' + filename
                
                # Octave execution
                mean_err_oct, std_error_oct, mean_nSV_oct, std_nSV_oct, mean_time_oct, std_time_oct = octave.feval('./LIBOL-0.3/demo.m', 'bc',algorithm,path_octave,'libsvm','c', nout = 6)
                
                # Python execution
                result_python = run('bc', algorithm, path, 'libsvm', print_results = False, test_parameters = True)
                mean_err_py  = result_python[0]
                mean_nSV_py  = result_python[1]
                mean_time_py = result_python[2]
                
                # Test for similar performance
                error_diff = abs(mean_err_oct - mean_err_py) 
                self.assertLessEqual(error_diff, 6*std_error_oct)
                
                # Test for similar execution time (python takes at most 1.5 more time as octave)
                self.assertLessEqual(mean_time_py, 1.5*mean_time_oct)

    # Test for similar performance and execution times for binary classification algorithms with random dataset sampling
    def test_mc(self):
        
        # Test all common algorithmstest_bc
        for algorithm in self.common_algorithms_mc:
            print("Testing ", algorithm ," (mc)")
            
            # Test all files in directory
            for filename in self.mc_files:
                
                
                path = self.mc_test_dir + '/' + filename
                path_octave = '/test/mc/' + filename
                
                # Octave execution
                mean_err_oct, std_error_oct, mean_nSV_oct, std_nSV_oct, mean_time_oct, std_time_oct = octave.feval('./LIBOL-0.3/demo.m', 'mc',algorithm,path_octave,'libsvm', nout = 6)
                
                # Python execution
                result_python = run('mc', algorithm, path, 'libsvm', print_results = False, test_parameters = True)
                mean_err_py  = result_python[0]
                mean_nSV_py  = result_python[1]
                mean_time_py = result_python[2]
                
                # Test for similar performance
                error_diff = abs(mean_err_oct - mean_err_py) 
                self.assertLessEqual(error_diff, 6*std_error_oct)
                
                # Test for better execution time
                self.assertLessEqual(mean_time_py, mean_time_oct)

    # Test Kernel Perceptron and Kernel OGD functionality
    def test_kernel_algorithms(self):
        
        # Test for each Kernel Algorithm
        for algorithm in self.OGD_algorithms:
            print("Testing ", algorithm )
            
            # Test all files in directory
            for filename in self.bc_files:

                path = self.bc_test_dir + '/' + filename
                result_python = run('bc', algorithm, path, 'libsvm', print_results = False, test_parameters = True)
    
    # Test OGD with different types of loss
    def test_OGD_loss_types(self):
        
        # Test for each Kernel Algorithm
        for algorithm in self.OGD_algorithms:
            print("Testing Loss types in ", algorithm )
            
            # Test all files in directory
            for filename in self.bc_files:
                
                path = self.bc_test_dir + '/' + filename
                
                # Test for different loss types
                for loss_type in range(4):
                    result_python = run('bc', algorithm, path, 'libsvm', print_results = False, test_parameters = True, loss_type = loss_type)

    
    # Test functionality of bc algorithm comparison
    def test_compare_bc(self):
        
        print("Testing BC Comparison")
        
        # Test all files in directory
        for filename in self.bc_files:

            path = self.bc_test_dir + '/' + filename
            result_python = compare('bc', path, 'libsvm', 1, print_results = False, showPlot = False)
        
        
    def test_compare_mc(self):
        
        print("Testing MC Comparison")
        
        # Test all files in directory
        for filename in self.mc_files:
            
            path = self.mc_test_dir + '/' + filename
            result_python = compare('mc', path, 'libsvm', 1, print_results = False, showPlot = False)

if __name__ == '__main__':
    unittest.main()
