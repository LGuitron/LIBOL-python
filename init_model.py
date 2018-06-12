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
                self.bias   = options.bias
            
            elif (UPmethod == 'PA1' or UPmethod == 'PA2'):
                self.bias   = options.bias
                self.C = options.C

            elif (UPmethod == 'ALMA'):
                self.bias   = options.bias
                self.C     = options.C
                self.alpha = options.eta
                self.p     = options.p           
                self.C     = options.C
                self.k_AL  = 1
                
            elif (UPmethod == 'OGD'):
                self.bias   = options.bias
                self.t         = options.t            # iteration number
                self.loss_type = options.loss_type    # loss type
                self.C         = options.C
            
            elif (UPmethod == 'CW'):
                self.bias   = options.bias
                self.Sigma = options.a*np.identity(d)
                self.eta   = options.eta
                self.phi   = norm.ppf(self.eta)
            
            elif(UPmethod =='AROW'):
                self.bias   = options.bias
                self.r     = options.C                         # parameter of AROW
                if(self.bias):
                    self.Sigma = options.a*np.identity(d+1)    # parameter of AROW
                else:
                    self.Sigma = options.a*np.identity(d)      # parameter of AROW
                
            elif(UPmethod =='SOP'):
                self.bias   = options.bias
                
                if(self.bias):
                    self.Sigma = options.a*np.identity(d+1)    # parameter of SOP
                else:
                    self.Sigma = options.a*np.identity(d)      # parameter of SOP

            elif (UPmethod == 'IELLIP'):
                self.bias   = options.bias
                self.b     = options.b
                self.c_t   = options.IELLIP_c
                self.Sigma = options.a*np.identity(d)
            
            elif (UPmethod == 'SCW'or UPmethod=='SCW2'):
                self.bias   = options.bias

                if(self.bias):
                    self.Sigma = options.a*np.identity(d+1)    # parameter of SCW
                else:
                    self.Sigma = options.a*np.identity(d)      # parameter of SCW
                    
                self.C     = options.C;
                self.eta   = options.eta;
                self.phi   = norm.ppf(self.eta)             # should use the inverse of normal function
                
            elif (UPmethod == 'NAROW'):
                self.bias   = options.bias
                self.b     = options.C                         # parameter of NAROW
                if(self.bias):
                    self.Sigma = options.a*np.identity(d+1)    # parameter of NAROW
                else:
                    self.Sigma = options.a*np.identity(d)      # parameter of NAROW
        
            elif (UPmethod == 'NHERD'):
                self.bias   = options.bias
                self.gamma = 1/options.C;
                self.Sigma = options.a*np.identity(d) # parameter of NAROW
                
            elif(UPmethod=='NEW_ALGORITHM'):
                # initialize the parameters of your algorithm...
                self.bias   = options.bias
        
            else:
                print('Unknown method.')
        
        elif (options.task_type == 'mc'):
            self.task_type = 'mc';
            self.W = np.zeros((int(nb_class),d))
            self.nb_class = nb_class;
            
            if (UPmethod == 'M_PERCEPTRONM' or UPmethod =='M_ROMMA' or UPmethod == 'M_AROMMA'):
                self.bias   = options.bias

            elif (UPmethod == 'M_PERCEPTRONU' or UPmethod == 'M_PERCEPTRONS'):
                self.bias   = options.bias
            
            elif (UPmethod == 'M_PA1' or UPmethod == 'M_PA2' or UPmethod =='M_PA'):
                self.bias   = options.bias
                self.C = options.C
                
            elif (UPmethod == 'M_OGD'):
                self.bias   = options.bias
                self.C   = options.C;          # learning rate parameter
                self.t   = options.t;          # iteration number
            
            elif (UPmethod == 'M_CW'):
                self.bias   = options.bias
                self.Sigma = options.a*np.identity(d)
                self.eta   = options.eta
                self.phi   = norm.ppf(self.eta)
        
            elif(UPmethod =='M_AROW'):
                self.bias   = options.bias
                self.r     = options.C                     # parameter of AROW
                self.Sigma = options.a*np.identity(d)      # parameter of AROW
                
            
            elif (UPmethod == 'M_SCW1'or UPmethod=='M_SCW2'):
                self.bias   = options.bias
                self.Sigma = options.a*np.identity(d);
                self.C     = options.C;
                self.phi   = norm.ppf(options.eta)
                
            elif(UPmethod=='NEW_ALGORITHM'):
                # initialize the parameters of your algorithm...
                self.bias   = options.bias
        
            else:
                print('Unknown method.')
