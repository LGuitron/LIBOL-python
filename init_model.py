import numpy as np
from scipy.stats import invgauss

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
            
            if (UPmethod == 'PERCEPTRON' or UPmethod =='ROMMA' or UPmethod == 'AROMMA' or 
                UPmethod =='PERCEPTRON_C' or UPmethod =='PA' or UPmethod =='PA_C' or 
                UPmethod =='ROMMA_C' or  UPmethod =='AROMMA_C'):
                pass
            
            elif (UPmethod == 'PA1' or UPmethod == 'PA2' or UPmethod == 'PA1_C' or UPmethod == 'PA2_C'):
                self.C = options.C

            elif (UPmethod == 'ALMA' or UPmethod == 'ALMA_C'):
                self.C     = options.C
                self.alpha = options.eta
                self.p     = options.p           
                self.C     = options.C
                self.k_AL  = 1
                
            elif (UPmethod == 'OGD' or UPmethod == 'OGD_C'):
                self.t         = options.t            # iteration number
                self.loss_type = options.loss_type    # loss type
                self.C         = options.C
            
            elif (UPmethod == 'CW' or UPmethod == 'CW_C'):
                self.Sigma = options.a*np.identity(d)
                self.eta   = options.eta
                #self.phi   = norminv(options.eta,0,1)  # should use the inverse of normal function
                self.phi   = invgauss.cdf(vals, mu)     # should use the inverse of normal function
        
            elif(UPmethod =='AROW' or UPmethod =='AROW_C'):
                self.r     = options.C                     # parameter of AROW
                self.Sigma = options.a*np.identity(d)      # parameter of AROW
                
            elif(UPmethod =='SOP' or UPmethod =='SOP_C'):
                self.Sigma = options.a*np.identity(d)      # parameter of SOP
        
            elif (UPmethod == 'IELLIP'or UPmethod=='IELLIP_C'):
                self.b     = options.b
                self.c_t   = options.IELLIP_c
                self.Sigma = options.a*np.identity(d)
            
            elif (UPmethod == 'SCW'or UPmethod=='SCW2' or UPmethod=='SCW_C' or UPmethod=='SCW2_C'):
                self.Sigma = options.a*eye(d);
                self.C     = options.C;
                #self.eta   = options.eta;
                #self.phi   = norminv(options.eta,0,1);     # should use the inverse of normal function
                self.phi   = invgauss.cdf(vals, mu)         # should use the inverse of normal function
                
            elif (UPmethod == 'NAROW'or UPmethod=='NAROW_C'):
                self.b     = options.C                     # parameter of NAROW
                self.Sigma = options.a*np.identity(d)      # parameter of NAROW
        
            elif (UPmethod == 'NHERD'or UPmethod=='NHERD_C'):
                self.gamma = 1/options.C;
                self.Sigma = options.a*np.identity(d) # parameter of NAROW
                
            elif(UPmethod=='NEW_ALGORITHM' or UPmethod=='NEW_ALGORITHM_C'):
                # initialize the parameters of your algorithm...
                pass
        
            else:
                print('Unknown method.')
        
        elif (options.task_type == 'mc'):
            self.task_type = 'mc';
            self.W = zeros(nb_class,d);
            self.nb_class = nb_class;
            
            if (UPmethod == 'M_PERCEPTRONM' or UPmethod =='M_ROMMA' or UPmethod == 'M_AROMMA' or 
                UPmethod =='M_PERCEPTRONM_C'  or UPmethod =='M_ROMMA_C' or  UPmethod =='M_AROMMA_C'):
                pass

            elif (UPmethod == 'M_PERCEPTRONU' or UPmethod =='M_PERCEPTRONU_C' or UPmethod == 'M_PERCEPTRONS' or UPmethod =='M_PERCEPTRONS_C'):
                pass
            
            elif (UPmethod == 'M_PA1' or UPmethod == 'M_PA2' or UPmethod == 'M_PA1_C' or UPmethod == 'M_PA2_C' or UPmethod =='M_PA' or UPmethod =='M_PA_C'):
                self.C = options.C
                
            elif (UPmethod == 'M_OGD' or UPmethod == 'M_OGD_C'):
                self.C   = options.C;          # learning rate parameter
                self.t   = options.t;          # iteration number
            
            elif (UPmethod == 'M_CW' or UPmethod == 'M_CW_C'):
                self.Sigma = options.a*np.identity(d)
                self.eta   = options.eta
                #self.phi   = norminv(options.eta,0,1)  # should use the inverse of normal function
                self.phi   = invgauss.cdf(vals, mu)     # should use the inverse of normal function
        
            elif(UPmethod =='M_AROW' or UPmethod =='M_AROW_C'):
                self.r     = options.C                     # parameter of AROW
                self.Sigma = options.a*np.identity(d)      # parameter of AROW
                
            
            elif (UPmethod == 'M_SCW1'or UPmethod=='M_SCW2' or UPmethod=='M_SCW_C' or UPmethod=='M_SCW2_C'):
                self.Sigma = options.a*eye(d);
                self.C     = options.C;
                #self.eta   = options.eta;
                #self.phi   = norminv(options.eta,0,1);     # should use the inverse of normal function
                self.phi   = invgauss.cdf(vals, mu)         # should use the inverse of normal function
                
            elif(UPmethod=='NEW_ALGORITHM' or UPmethod=='NEW_ALGORITHM_C'):
                # initialize the parameters of your algorithm...
                pass
        
            else:
                print('Unknown method.')
