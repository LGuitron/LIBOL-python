from compare import compare
from oct2py import octave
import os
import unittest

# Tests for comparing LIBOL (Python) and LIBOL (MATLAB)
class LIBOLtests(unittest.TestCase):

    def setUp(self):
        
        self.bc_test_dir   = './data/test/bc'
        self.bc_files      = os.listdir(self.bc_test_dir)
        self.bc_algorithms = ['Perceptron','Kernel_Perceptron', 'PA','PA1','PA2','OGD','Kernel_OGD','SOP','CW','SCW','SCW2','AROW','NAROW'] 
        
        # Download Version 3 of LIBOL for MATLAB
        if not (os.path.isdir('./LIBOL-0.3')):
            print("\n____________________________")
            print("Downloading LIBOL for MATLAB")
            print("____________________________\n")
            os.system('wget https://github.com/LIBOL/LIBOL/archive/v0.3.zip')
            os.system('unzip v0.3.zip')
            os.system('rm v0.3.zip')
            
        # Compile octave modules
        print("\n____________________________")
        print("Compiling modules for Octave")
        print("____________________________\n")
        octave.run('./LIBOL-0.3/make.m')
        
        # Run tests for all binary classification datasets in LIBOL MATLAB
        print("Running binary classification algorithms on LIBOL Matlab ...")
        for filename in self.bc_files:
            path = self.bc_test_dir + '/' + filename
            path_octave = '/test/bc/' + filename
            res  = octave.feval('./LIBOL-0.3/demo.m', 'bc','Perceptron',path_octave,'libsvm')

    # Test performance against LIBOL for matlab
    def test_bc_performance(self):
        #libol_py_results = compare('bc','./data/bc/breast_cancer_scale.txt','libsvm', showPlot = False)
        #octave.run("cos(pi / 3)")
        r = octave.feval("cos",3.1415926538/3)
        #print(libol_py_results)
        #self.assertEqual(1, 1)
        
    def tearDown(self):
        # Remove LIBOL release for MATLAB
        if (os.path.isdir('./LIBOL-0.3')):
            os.system('rm -r ./LIBOL-0.3')
    
if __name__ == '__main__':
    unittest.main()
