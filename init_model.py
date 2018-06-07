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
            self.w = np.zeros((1,d));       #weight vector of classifier
            
            if (UPmethod == 'PERCEPTRON' or UPmethod =='ROMMA' or UPmethod == 'AROMMA' or UPmethod =='PA'):
                pass
            
            elif (UPmethod == 'PA1' or UPmethod == 'PA2'):
                self.C = options.C

            elif (UPmethod == 'ALMA'):
                self.C     = options.C
                self.alpha = options.eta
                self.p     = options.p           
                self.C     = options.C
                self.k_AL  = 1
                
            elif (UPmethod == 'OGD'):
                self.t         = options.t            # iteration number
                self.loss_type = options.loss_type    # loss type
                self.C         = options.C
            
            elif (UPmethod == 'CW'):
                self.Sigma = options.a*np.identity(d)
                self.eta   = options.eta
                #self.phi   = norminv(options.eta,0,1)      # should use the inverse of normal function
                #self.phi   = invgauss.cdf(self.eta, 0)     # should use the inverse of normal function
                self.phi   = norm.ppf(self.eta)
            
            elif(UPmethod =='AROW'):
                self.r     = options.C                     # parameter of AROW
                self.Sigma = options.a*np.identity(d)      # parameter of AROW
                
            elif(UPmethod =='SOP'):
                self.Sigma = options.a*np.identity(d)      # parameter of SOP
        
            elif (UPmethod == 'IELLIP'):
                self.b     = options.b
                self.c_t   = options.IELLIP_c
                self.Sigma = options.a*np.identity(d)
            
            elif (UPmethod == 'SCW'or UPmethod=='SCW2'):
                self.Sigma = options.a*np.identity(d);
                self.C     = options.C;
                self.eta   = options.eta;
                #self.phi   = norminv(options.eta,0,1);     # should use the inverse of normal function
                self.phi   = norm.ppf(self.eta)             # should use the inverse of normal function
                
            elif (UPmethod == 'NAROW'):
                self.b     = options.C                     # parameter of NAROW
                self.Sigma = options.a*np.identity(d)      # parameter of NAROW
        
            elif (UPmethod == 'NHERD'):
                self.gamma = 1/options.C;
                self.Sigma = options.a*np.identity(d) # parameter of NAROW
                
            elif(UPmethod=='NEW_ALGORITHM'):
                # initialize the parameters of your algorithm...
                pass
        
            else:
                print('Unknown method.')
        
        elif (options.task_type == 'mc'):
            self.task_type = 'mc';
            self.W = np.zeros((int(nb_class),d))
            self.nb_class = nb_class;
            
            if (UPmethod == 'M_PERCEPTRONM' or UPmethod =='M_ROMMA' or UPmethod == 'M_AROMMA'):
                pass

            elif (UPmethod == 'M_PERCEPTRONU' or UPmethod == 'M_PERCEPTRONS'):
                pass
            
            elif (UPmethod == 'M_PA1' or UPmethod == 'M_PA2' or UPmethod =='M_PA'):
                self.C = options.C
                
            elif (UPmethod == 'M_OGD'):
                self.C   = options.C;          # learning rate parameter
                self.t   = options.t;          # iteration number
            
            elif (UPmethod == 'M_CW'):
                self.Sigma = options.a*np.identity(d)
                self.eta   = options.eta
                #self.phi   = norminv(options.eta,0,1)  # should use the inverse of normal function
                #self.phi   = invgauss.cdf(vals, mu)     # should use the inverse of normal function
                self.phi   = norm.ppf(self.eta)
        
            elif(UPmethod =='M_AROW'):
                self.r     = options.C                     # parameter of AROW
                self.Sigma = options.a*np.identity(d)      # parameter of AROW
                
            
            elif (UPmethod == 'M_SCW1'or UPmethod=='M_SCW2'):
                self.Sigma = options.a*np.identity(d);
                self.C     = options.C;
                #self.eta   = options.eta;
                #self.phi   = norminv(options.eta,0,1);     # should use the inverse of normal function
                #self.phi   = invgauss.cdf(vals, mu)         # should use the inverse of normal function
                self.phi   = norm.ppf(options.eta)
                
            elif(UPmethod=='NEW_ALGORITHM'):
                # initialize the parameters of your algorithm...
                pass
        
            else:
                print('Unknown method.')
