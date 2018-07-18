import numpy as np
from regularizers.Regularizer import L0
from regularizers.Regularizer import TGD
from kernels.Kernels import gaussian_kernel
from sklearn import preprocessing as pp
import numpy as np

class Options:

    def __init__(self, method,n,task_type):
        # init_options: initialize the options for each method
        #--------------------------------------------------------------------------
        # INPUT:
        #       method:            method name
        #       n:                 number of training instances in the database
        #       task_type:         type of task (bc or mc)
        #       bias:              Add bias weight in the algorithms
        #       regularization     Apply specified regularization in algorithms
        
        self.method = method
        self.t_tick = round(n/15)    #10
        self.task_type = task_type
        self.id_list = np.random.permutation(n)
        UPmethod = method.upper()
    
        '''
        
        Initial Parameter values can be modified below
        
        '''
        
        # Options for Binary Classification algorithms
        if (task_type == 'bc'):
            
            if (UPmethod == 'PERCEPTRON' or UPmethod =='PA'):
                self.bias            = True
                self.p_kernel_degree = 1                            # Input Preprocesing to be applied 
            
            elif (UPmethod == 'GAUSSIAN_KERNEL_PERCEPTRON'):
                self.max_sv       = 100                             # Number of instances to keep for kernel approach                 
                self.kernel       = gaussian_kernel                 # Kernel method
                self.sigma        = 1                               # Hyperparameter to use in gaussian_kernel
            
            elif (UPmethod =='PA1' or UPmethod =='PA2'):
                self.bias            = True
                self.p_kernel_degree = 1 
                self.C               = 1

            elif (UPmethod == 'OGD'):
                self.bias            = True
                self.p_kernel_degree = 1 
                self.loss_type       = 1                              # type of loss (0, 0-1 loss, 1 - hinge, 2-log, 3-square )
                self.C               = 1
                #self.regularizer    = None                           # No regularizer
                #self.regularizer    = L0 (theta = 1.5)               # Coefficient rounding regularizer
                self.regularizer     = TGD(theta = 1.5, g = 0.025)    # L1 regularizer (gradual decrease of small coefficients)
            
            elif (UPmethod == 'GAUSSIAN_KERNEL_OGD'):
                self.loss_type    = 1                           # type of loss (0, 0-1 loss, 1 - hinge, 2-log, 3-square )
                self.C            = 1
                self.max_sv       = 100                          # Number of instances to keep for kernel approach                 
                self.kernel       = gaussian_kernel              # Kernel method
                self.sigma        = 1                            # Hyperparameter to use in gaussian_kernel
            
                
            elif (UPmethod == 'CW'):
                self.bias            = True
                self.p_kernel_degree = 1
                self.eta             = 0.7  # in \eta in [0.5,1]
                self.a               = 1

            elif (UPmethod == 'AROW'):
                self.bias            = True
                self.p_kernel_degree = 1 
                self.C               = 1      # i.e., parameter r
                self.a               = 1      # default

            elif (UPmethod == 'SOP'):
                self.bias            = True
                self.p_kernel_degree = 1 
                self.a               = 1

            elif (UPmethod == 'SCW'):
                self.bias            = True
                self.p_kernel_degree = 1 
                self.eta             = 0.75
                self.C               = 1
                self.a               = 1

            elif (UPmethod == 'SCW2'):
                self.bias            = True
                self.p_kernel_degree = 1 
                self.eta             = 0.9
                self.C               = 1
                self.a               = 1

            elif (UPmethod == 'NAROW'):
                self.bias            = True
                self.p_kernel_degree = 1 
                self.C               = 1         #i.e., parameter r
                self.a               = 1

            elif (UPmethod == 'NEW_ALGORITHM'):
                pass
            
            else:
                print('Unknown method.')
        
        # Options for Multiclass Classification algorithms
        elif (task_type == 'mc'):
            
            if (UPmethod == 'M_PERCEPTRONM' or UPmethod == 'M_PERCEPTRONU' or UPmethod == 'M_PERCEPTRONS'):
                self.bias            = True
                self.p_kernel_degree = 1

            elif (UPmethod == 'M_OGD'):
                self.bias            = True
                self.p_kernel_degree = 1 
                self.C = 1
                #self.regularizer = None                           # No regularizer
                #self.regularizer = L0 (theta = 1.5)               # Coefficient rounding regularizer
                self.regularizer  = TGD(theta = 1.5, g = 0.025)    # L1 regularizer (gradual decrease of small coefficients)

            elif (UPmethod == 'M_PA' or UPmethod == 'M_PA1' or UPmethod =='M_PA2'):
                self.bias            = True
                self.p_kernel_degree = 2 
                self.C = 1

            elif (UPmethod == 'M_CW'):
                self.bias            = True
                self.p_kernel_degree = 1 
                self.eta             = 0.75  # in \eta in [0.5,1]
                self.a               = 1

            elif (UPmethod == 'M_SCW1' or UPmethod =='M_SCW2'):
                self.bias            = True
                self.p_kernel_degree = 1 
                self.eta             = 0.75
                self.C               = 1
                self.a               = 1

            elif (UPmethod == 'M_AROW'):
                self.bias            = True
                self.p_kernel_degree = 1 
                self.C               = 1      # i.e., parameter r
                self.a               = 1      # default

            elif (UPmethod == 'NEW_ALGORITHM'):
                pass
    
        '''
        
        Hyperparameters tuning ranges can be modified below
        
        '''
        
        self.range_C   = 2**np.arange(-4.0,4.0,1.0)
        self.range_eta = np.arange(0.55,0.95,0.05)
        self.range_b   = np.arange(0.1,0.9,0.1)
        self.range_p   = np.arange(2,10,2)
