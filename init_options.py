import numpy as np
from regularizers.Regularizer import L0
from regularizers.Regularizer import TGD
from kernels.Kernels import gaussian_kernel
import numpy as np

class Options:

    def __init__(self, method,n,task_type, bias = True, regularization = True):
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

        # Variable to determine if bias parameter will be used in the model
        self.bias = bias
        
        
        if (task_type == 'bc'):
            
            if (UPmethod == 'PERCEPTRON' or UPmethod =='PA'):
                pass
            
            elif (UPmethod == 'KERNEL_PERCEPTRON'):
                self.max_sv       = 100                          # Number of instances to keep for kernel approach                 
                self.kernel       = gaussian_kernel              # Kernel method
                self.sigma        = 1                            # Hyperparameter to use in gaussian_kernel
                self.index        = 0                            # Index for replacing SV when using Budget MAintenance
            
            elif (UPmethod =='PA1' or UPmethod =='PA2'):
                self.C = 1

            elif (UPmethod == 'OGD'):
                self.t            = 1                              # iteration no, learning rate eta_t = 1/sqrt(t)
                self.loss_type    = 1                              # type of loss (0, 0-1 loss, 1 - hinge, 2-log, 3-square )
                self.C            = 1
                
                if not regularization:
                    self.regularizer = None                            # No regularizer
                
                else:
                    #self.regularizer = L0 (theta = 1.5)               # Coefficient rounding regularizer
                    self.regularizer  = TGD(theta = 1.5, g = 0.025)   # L1 regularizer (gradual decrease of small coefficients)
            
            elif (UPmethod == 'KERNEL_OGD'):
                self.t            = 1                            # iteration no, learning rate eta_t = 1/sqrt(t)
                self.loss_type    = 1                            # type of loss (0, 0-1 loss, 1 - hinge, 2-log, 3-square )
                self.C            = 1

                self.max_sv       = 100                          # Number of instances to keep for kernel approach                 
                self.kernel       = gaussian_kernel              # Kernel method
                self.sigma        = 1                            # Hyperparameter to use in gaussian_kernel
                self.index        = 0                            # Index for replacing SV when using Budget MAintenance
            
                
            elif (UPmethod == 'CW'):
                self.eta = 0.7  # in \eta in [0.5,1]
                self.a   = 1

            elif (UPmethod == 'AROW'):
                self.C = 1      # i.e., parameter r
                self.a = 1      # default

            elif (UPmethod == 'SOP'):
                self.a = 1

            elif (UPmethod == 'SCW'):
                self.eta = 0.75
                self.C   = 1
                self.a   = 1

            elif (UPmethod == 'SCW2'):
                self.eta = 0.9
                self.C   = 1
                self.a   = 1

            elif (UPmethod == 'NAROW'):
                self.C = 1         #i.e., parameter r
                self.NAROW_b = 1
                self.a = 1

            elif (UPmethod == 'NEW_ALGORITHM'):
                pass
            
            else:
                print('Unknown method.')

        elif (task_type == 'mc'):
            
            if (UPmethod == 'M_PERCEPTRONM' or UPmethod == 'M_PERCEPTRONU' or UPmethod == 'M_PERCEPTRONS'):
                pass

            elif (UPmethod == 'M_OGD'):
                self.t = 1                                             # iteration no, learning rate eta_t = 1/sqrt(t)
                self.C = 1
                
                if not regularization:
                    self.regularizer = None                            # No regularizer
                
                else:
                    #self.regularizer = L0 (theta = 1.5)               # Coefficient rounding regularizer
                    self.regularizer  = TGD(theta = 1.5, g = 0.025)    # L1 regularizer (gradual decrease of small coefficients)

            elif (UPmethod == 'M_PA' or UPmethod == 'M_PA1' or UPmethod =='M_PA2'):
                self.C = 1

            elif (UPmethod == 'M_CW'):
                self.eta = 0.75  # in \eta in [0.5,1]
                self.a   = 1

            elif (UPmethod == 'M_SCW1' or UPmethod =='M_SCW2'):
                self.eta = 0.75
                self.C   = 1
                self.a   = 1

            elif (UPmethod == 'M_AROW'):
                self.C = 1      # i.e., parameter r
                self.a = 1      # default

            elif (UPmethod == 'NEW_ALGORITHM'):
                pass
