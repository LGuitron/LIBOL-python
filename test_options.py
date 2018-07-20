import numpy as np
from regularizers.Regularizer import L0
from regularizers.Regularizer import TGD
from kernels.Kernels import gaussian_kernel
import numpy as np

class TestOptions:

    def __init__(self, method,n,task_type, loss_type = 1):
        # init_options: initialize the options for each method
        #--------------------------------------------------------------------------
        # INPUT:
        #       method:            method name
        #       n:                 number of training instances in the database
        #       task_type:         type of task (bc or mc)
        #       bias:              Add bias weight in the algorithms
        #       regularization     Apply specified regularization in algorithms
        
        self.method    = method
        self.t_tick    = round(n/15)    #10
        self.task_type = task_type
        self.id_list   = np.random.permutation(n)
        self.loss_type = loss_type                  # Loss type for OGD passed in constructor 
        self.p_kernel_degree = 1  

        UPmethod = method.upper()
        
        if (task_type == 'bc'):
            
            if (UPmethod == 'PERCEPTRON' or UPmethod =='PA'):
                self.bias            = False                      # Input Preprocesing to be applied 
            
            elif (UPmethod == 'GAUSSIAN_KERNEL_PERCEPTRON'):
                self.max_sv       = 100                          # Number of instances to keep for kernel approach                 
                self.kernel       = gaussian_kernel              # Kernel method
                self.sigma        = 1                            # Hyperparameter to use in gaussian_kernel
            
            elif (UPmethod =='PA1' or UPmethod =='PA2'):
                self.bias         = False
                self.C = 1

            elif (UPmethod == 'OGD'):
                self.bias         = False
                self.C            = 1
                self.regularizer = None                            # No regularizer
            
            elif (UPmethod == 'GAUSSIAN_KERNEL_OGD'):
                self.C            = 1
                self.max_sv       = 100                          # Number of instances to keep for kernel approach                 
                self.kernel       = gaussian_kernel              # Kernel method
                self.sigma        = 1                            # Hyperparameter to use in gaussian_kernel
            
                
            elif (UPmethod == 'CW'):
                self.bias         = False
                self.eta          = 0.7  # in \eta in [0.5,1]
                self.a            = 1

            elif (UPmethod == 'AROW'):
                self.bias         = False
                self.C            = 1      # i.e., parameter r
                self.a            = 1      # default

            elif (UPmethod == 'SOP'):
                self.bias         = False
                self.a            = 1

            elif (UPmethod == 'SCW'):
                self.bias         = False
                self.eta          = 0.75
                self.C            = 1
                self.a            = 1

            elif (UPmethod == 'SCW2'):
                self.bias         = False
                self.eta         = 0.9
                self.C           = 1
                self.a           = 1

            elif (UPmethod == 'NAROW'):
                self.bias         = False
                self.C            = 1         #i.e., parameter r
                self.NAROW_b      = 1
                self.a            = 1

            elif (UPmethod == 'NEW_ALGORITHM'):
                pass
            
            else:
                print('Unknown method bc test_options.')

        elif (task_type == 'mc'):
            
            if (UPmethod == 'M_PERCEPTRONM' or UPmethod == 'M_PERCEPTRONU' or UPmethod == 'M_PERCEPTRONS'):
                self.bias         = False

            elif (UPmethod == 'M_OGD'):
                self.bias         = False
                self.C            = 1
                self.regularizer  = None                            # No regularizer

            elif (UPmethod == 'M_PA' or UPmethod == 'M_PA1' or UPmethod =='M_PA2'):
                self.bias         = False
                self.C            = 1

            elif (UPmethod == 'M_CW'):
                self.bias         = False
                self.eta          = 0.75  # in \eta in [0.5,1]
                self.a            = 1

            elif (UPmethod == 'M_SCW1' or UPmethod =='M_SCW2'):
                self.bias         = False
                self.eta          = 0.75
                self.C            = 1
                self.a            = 1

            elif (UPmethod == 'M_AROW'):
                self.bias         = False
                self.C            = 1      # i.e., parameter r
                self.a            = 1      # default

            elif (UPmethod == 'NEW_ALGORITHM'):
                pass
        
        self.range_C   = 2**np.arange(-4.0,4.0,1.0)
        self.range_eta = np.arange(0.55,0.95,0.05)
        self.range_b   = np.arange(0.1,0.9,0.1)
        self.range_p   = np.arange(2,10,2)
