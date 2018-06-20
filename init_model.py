import numpy as np
from scipy.stats import norm

class Model:

    def __init__(self, options, d, nb_class):
    # INPUT:
    #  options:     method name and setting
    #        d:     data dimensionality
    # nb_class:     number of class labels
        
        UPmethod = options.method.upper()
        if (options.task_type == 'bc'):
            self.task_type = 'bc'
            
            # Weight vector with bias term (at the beginning)
            if(options.bias):
                self.w = np.zeros((1,d+1))
                
            # No bias term
            else:
                self.w = np.zeros((1,d))
            
            if (UPmethod == 'PERCEPTRON' or UPmethod =='ROMMA' or UPmethod == 'AROMMA' or UPmethod =='PA'):
                self.bias        = options.bias
                self.regularizer = options.regularizer
                
            elif (UPmethod == 'PA1' or UPmethod == 'PA2'):
                self.bias   = options.bias
                self.C = options.C
                self.regularizer = options.regularizer

            elif (UPmethod == 'ALMA'):
                self.bias   = options.bias
                self.C     = options.C
                self.alpha = options.eta
                self.p     = options.p           
                self.C     = options.C
                self.k_AL  = 1
                self.regularizer = options.regularizer
                
            elif (UPmethod == 'OGD'):
                self.bias        = options.bias
                self.t           = options.t            # iteration number
                self.loss_type   = options.loss_type    # loss type
                self.C           = options.C
                self.regularizer = options.regularizer

            elif (UPmethod == 'CW'):
                self.bias   = options.bias
                
                if(self.bias):
                    self.Sigma = options.a*np.identity(d+1)    # parameter of CW
                else:
                    self.Sigma = options.a*np.identity(d)      # parameter of CW
                    
                self.eta   = options.eta
                self.phi   = norm.ppf(self.eta)
                self.regularizer = options.regularizer
            
            elif(UPmethod =='AROW'):
                self.bias   = options.bias
                self.r     = options.C                         # parameter of AROW
                if(self.bias):
                    self.Sigma = options.a*np.identity(d+1)    # parameter of AROW
                else:
                    self.Sigma = options.a*np.identity(d)      # parameter of AROW
                self.regularizer = options.regularizer
                
            elif(UPmethod =='SOP'):
                self.bias   = options.bias
                
                if(self.bias):
                    self.Sigma = options.a*np.identity(d+1)    # parameter of SOP
                else:
                    self.Sigma = options.a*np.identity(d)      # parameter of SOP
                self.regularizer = options.regularizer
                
            elif (UPmethod == 'IELLIP'):
                self.bias   = options.bias
                self.b     = options.b
                self.c_t   = options.IELLIP_c
                self.Sigma = options.a*np.identity(d)
                self.regularizer = options.regularizer
            
            elif (UPmethod == 'SCW'or UPmethod=='SCW2'):
                self.bias   = options.bias

                if(self.bias):
                    self.Sigma = options.a*np.identity(d+1)    # parameter of SCW
                else:
                    self.Sigma = options.a*np.identity(d)      # parameter of SCW
                    
                self.C     = options.C;
                self.eta   = options.eta;
                self.phi   = norm.ppf(self.eta)             # should use the inverse of normal function
                self.regularizer = options.regularizer
                
            elif (UPmethod == 'NAROW'):
                self.bias   = options.bias
                self.b     = options.C                         # parameter of NAROW
                if(self.bias):
                    self.Sigma = options.a*np.identity(d+1)    # parameter of NAROW
                else:
                    self.Sigma = options.a*np.identity(d)      # parameter of NAROW
                self.regularizer = options.regularizer
                
            elif (UPmethod == 'NHERD'):
                self.bias   = options.bias
                self.gamma = 1/options.C;
                self.Sigma = options.a*np.identity(d) # parameter of NAROW
                self.regularizer = options.regularizer
                
            elif(UPmethod=='NEW_ALGORITHM'):
                # initialize the parameters of your algorithm...
                self.bias   = options.bias
                self.regularizer = options.regularizer
            else:
                print('Unknown method.')
        
        elif (options.task_type == 'mc'):
            self.task_type = 'mc';
            self.nb_class = nb_class;
            
            # Bias term for each class
            if(options.bias):
                self.W = np.zeros((int(nb_class),d+1))

            # No bias terms
            else:
                self.W = np.zeros((int(nb_class),d))
            
            if (UPmethod == 'M_PERCEPTRONM' or UPmethod =='M_ROMMA' or UPmethod == 'M_AROMMA'):
                self.bias   = options.bias
                self.regularizer = options.regularizer
                
            elif (UPmethod == 'M_PERCEPTRONU' or UPmethod == 'M_PERCEPTRONS'):
                self.bias   = options.bias
                self.regularizer = options.regularizer
            
            elif (UPmethod == 'M_PA1' or UPmethod == 'M_PA2' or UPmethod =='M_PA'):
                self.bias   = options.bias
                self.C = options.C
                self.regularizer = options.regularizer
                
            elif (UPmethod == 'M_OGD'):
                self.bias   = options.bias
                self.C   = options.C;          # learning rate parameter
                self.t   = options.t;          # iteration number
                self.regularizer = options.regularizer
            
            elif (UPmethod == 'M_CW'):
                self.bias   = options.bias
                self.eta   = options.eta
                self.phi   = norm.ppf(self.eta)
                if(self.bias):
                    self.Sigma = options.a*np.identity(d+1)    # parameter of M_SCW
                else:
                    self.Sigma = options.a*np.identity(d)      # parameter of M_SCW
                self.regularizer = options.regularizer
        
            elif(UPmethod =='M_AROW'):
                self.bias   = options.bias
                self.r     = options.C                     # parameter of AROW
                if(self.bias):
                    self.Sigma = options.a*np.identity(d+1)    # parameter of M_AROW
                else:
                    self.Sigma = options.a*np.identity(d)      # parameter of M_AROW
                self.regularizer = options.regularizer
            
            elif (UPmethod == 'M_SCW1'or UPmethod=='M_SCW2'):
                self.bias   = options.bias
                self.C     = options.C;
                self.phi   = norm.ppf(options.eta)
                if(self.bias):
                    self.Sigma = options.a*np.identity(d+1)    # parameter of M_SCW
                else:
                    self.Sigma = options.a*np.identity(d)      # parameter of M_SCW
                self.regularizer = options.regularizer
                
            elif(UPmethod=='NEW_ALGORITHM'):
                # initialize the parameters of your algorithm...
                self.bias   = options.bias
                self.regularizer = options.regularizer
            else:
                print('Unknown method.')
