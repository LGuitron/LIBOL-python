import numpy as np
from scipy.stats import norm
from sklearn import preprocessing as pp

class Model:

    def __init__(self, options, d, nb_class):
    # INPUT:
    #  options:     method name and setting
    #        d:     data dimensionality
    # nb_class:     number of class labels

        UPmethod = options.method.upper()
        if (options.task_type == 'bc'):
            self.task_type = 'bc'

            # Don't add weight vector for kernel algorithms
            if (UPmethod == 'GAUSSIAN_KERNEL_PERCEPTRON' or UPmethod == 'GAUSSIAN_KERNEL_OGD'):
                pass

            
            # Set weight vector size depending on polynomial feature map
            elif (options.p_kernel_degree > 1):
                self.poly = pp.PolynomialFeatures(degree = options.p_kernel_degree , include_bias = options.bias)
                self.w    = np.zeros(self.poly.fit_transform(np.zeros((1,d))).shape) 

            # Dont add bias when explicitly specified
            elif (not options.bias):
                self.w = np.zeros((1,d))
            
            # Add bias otherwise
            else:
                self.w = np.zeros((1,d+1))
            
            if (UPmethod == 'PERCEPTRON' or UPmethod == 'PA'):
                self.bias            = options.bias
                self.p_kernel_degree = options.p_kernel_degree
                
            
            
            elif (UPmethod == 'GAUSSIAN_KERNEL_PERCEPTRON'):
                self.max_sv       = options.max_sv                # Number of instances to keep for kernel approach
                self.alpha        = np.zeros(self.max_sv)        # Weights corresponding to each of the support vectors
                self.SV           = np.zeros((self.max_sv,d))    # Support vector array with values of x
                self.sv_num       = 0                            # Number of support vectors added so far
                self.kernel       = options.kernel               # Kernel method to use
                self.sigma        = options.sigma                # Hyperparameter for gaussian kernel
                self.index        = 0                            # Index for budget maintenance
            
            elif (UPmethod == 'PA1' or UPmethod == 'PA2'):
                self.bias   = options.bias
                self.C = options.C
                
            elif (UPmethod == 'OGD'):
                self.bias        = options.bias
                self.t           = 1                    # iteration number
                self.loss_type   = options.loss_type    # loss type
                self.C           = options.C
                self.regularizer = options.regularizer
            
            elif (UPmethod == 'GAUSSIAN_KERNEL_OGD'):
                self.t           = 1                    # iteration number
                self.loss_type   = options.loss_type    # loss type
                self.C           = options.C

                self.max_sv       = options.max_sv               # Number of instances to keep for kernel approach
                self.alpha        = np.zeros(self.max_sv)        # Weights corresponding to each of the support vectors
                self.SV           = np.zeros((self.max_sv,d))    # Support vector array with values of x
                self.sv_num       = 0                            # Number of support vectors added so far
                self.kernel       = options.kernel               # Kernel method to use
                self.sigma        = options.sigma                # Hyperparameter for gaussian kernel
                self.index        = 0                            # Index for budget maintenance
                
            elif (UPmethod == 'CW'):
                self.bias   = options.bias
                
                if(self.bias):
                    self.Sigma = options.a*np.identity(d+1)    # parameter of CW
                else:
                    self.Sigma = options.a*np.identity(d)      # parameter of CW
                    
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
                
            elif(UPmethod=='NEW_ALGORITHM'):
                pass

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
            
            if (UPmethod == 'M_PERCEPTRONM' or UPmethod == 'M_PERCEPTRONU' or UPmethod == 'M_PERCEPTRONS'):
                self.bias   = options.bias
            
            elif (UPmethod == 'M_PA1' or UPmethod == 'M_PA2' or UPmethod =='M_PA'):
                self.bias   = options.bias
                self.C = options.C
                
            elif (UPmethod == 'M_OGD'):
                self.bias   = options.bias
                self.C   = options.C;          # learning rate parameter
                self.t   = 1;                  # iteration number
                self.regularizer = options.regularizer
            
            elif (UPmethod == 'M_CW'):
                self.bias   = options.bias
                self.eta   = options.eta
                self.phi   = norm.ppf(self.eta)
                if(self.bias):
                    self.Sigma = options.a*np.identity(d+1)    # parameter of M_SCW
                else:
                    self.Sigma = options.a*np.identity(d)      # parameter of M_SCW
        
            elif(UPmethod =='M_AROW'):
                self.bias   = options.bias
                self.r     = options.C                     # parameter of AROW
                if(self.bias):
                    self.Sigma = options.a*np.identity(d+1)    # parameter of M_AROW
                else:
                    self.Sigma = options.a*np.identity(d)      # parameter of M_AROW
            
            elif (UPmethod == 'M_SCW1'or UPmethod=='M_SCW2'):
                self.bias   = options.bias
                self.C     = options.C;
                self.phi   = norm.ppf(options.eta)
                if(self.bias):
                    self.Sigma = options.a*np.identity(d+1)    # parameter of M_SCW
                else:
                    self.Sigma = options.a*np.identity(d)      # parameter of M_SCW
                
            elif(UPmethod=='NEW_ALGORITHM'):
                pass

            else:
                print('Unknown method.')
