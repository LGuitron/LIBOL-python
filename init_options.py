import numpy as np
import imp
from regularizers.Regularizer import L0
from regularizers.Regularizer import TGD

class Options:

    def __init__(self, method,n,task_type):
        # init_options: initialize the options for each method
        #--------------------------------------------------------------------------
        # INPUT:
        #       method:     method name
        #       n:          number of training instances in the database
        #       task_type:  type of task (bc or mc)

        self.method = method
        self.t_tick = round(n/15)    #10
        self.task_type = task_type
        self.id_list = np.random.permutation(n)
        UPmethod = method.upper()

        if (task_type == 'bc'):
            
            if (UPmethod == 'PERCEPTRON' or UPmethod =='ROMMA' or UPmethod == 'AROMMA' or UPmethod =='PA'):
                self.bias         = True
                #self.regularizer = None                                             # No regularizer
                #self.regularizer = L0 (theta = 1.5, update_freq = 500)              # Coefficient rounding regularizer
                self.regularizer  = TGD(theta = 1.5, g = 0.3, update_freq = 500)     # L1 regularizer (gradual decrease of small coefficients)
            
            elif (UPmethod == 'WINNOW' or UPmethod =='PA1' or UPmethod =='PA2'):
                self.bias = True
                self.C = 1
            
            elif (UPmethod == 'ALMA'):
                self.bias = True
                self.eta = 0.9  # alpha(eta) \in (0,1]
                self.p   = 2
                self.C   = sqrt(2)

            elif (UPmethod == 'OGD'):
                self.bias         = True
                self.t            = 1                                                 # iteration no, learning rate eta_t = 1/sqrt(t)
                self.loss_type    = 1                                                 # type of loss (0, 0-1 loss, 1 - hinge, 2-log, 3-square )
                self.C            = 1
                #self.regularizer = None                                             # No regularizer
                #self.regularizer = L0 (theta = 1.5, update_freq = 500)              # Coefficient rounding regularizer
                self.regularizer = TGD(theta = 1.5, g = 0.5, update_freq = 3000)     # L1 regularizer (gradual decrease of small coefficients)
                
                
            elif (UPmethod == 'CW'):
                self.bias = True
                self.eta = 0.7  # in \eta in [0.5,1]
                self.a   = 1

            elif (UPmethod == 'AROW'):
                self.bias = True
                self.C = 1      # i.e., parameter r
                self.a = 1      # default
            
            
            elif (UPmethod == 'SOP'):
                self.bias = True
                self.a = 1

            elif (UPmethod == 'IELLIP'):
                self.bias = True
                self.b = 0.3
                self.IELLIP_c = 0.1   # c=0.5
                self.a = 1

            elif (UPmethod == 'SCW'):
                self.bias = True
                self.eta = 0.75
                self.C   = 1
                self.a   = 1
        
            elif (UPmethod == 'SCW2'):
                self.bias = True
                self.eta = 0.9
                self.C   = 1
                self.a   = 1

            elif (UPmethod == 'NAROW'):
                self.bias = True
                self.C = 1         #i.e., parameter r
                self.NAROW_b = 1
                self.a = 1

            elif (UPmethod == 'NHERD'):
                self.bias = True
                self.C = 1
                self.a = 1

            elif (UPmethod == 'NEW_ALGORITHM'):
                # initialie your parameters here...
                self.bias = True
            
            else:
                print('Unknown method.')

        elif (task_type == 'mc'):
            
            if (UPmethod == 'M_PERCEPTRONM' or UPmethod =='M_ROMMA' or UPmethod == 'M_AROMMA'):
                self.bias = True
            
            elif (UPmethod == 'M_PERCEPTRONU'):
                self.bias = True
            
            elif (UPmethod == 'M_PERCEPTRONS' ):
                self.bias = True
            
            elif (UPmethod == 'M_OGD'):
                self.bias = True
                self.t = 1      # iteration no, learning rate eta_t = 1/sqrt(t)
                self.C = 1

            elif (UPmethod == 'M_PA' or UPmethod == 'M_PA1' or UPmethod =='M_PA2'):
                self.bias = True
                self.C = 1

            elif (UPmethod == 'M_CW'):
                self.bias = True
                self.eta = 0.75  # in \eta in [0.5,1]
                self.a   = 1

            elif (UPmethod == 'M_SCW1' or UPmethod =='M_SCW2'):
                self.bias = True
                self.eta = 0.75
                self.C   = 1
                self.a   = 1

            elif (UPmethod == 'M_AROW'):
                self.bias = True
                self.C = 1      # i.e., parameter r
                self.a = 1      # default

            elif (UPmethod == 'NEW_ALGORITHM'):
                # initialie your parameters here...
                self.bias = True
