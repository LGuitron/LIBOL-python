import numpy as np
from  math import log, exp
def Kernel_OGD(y_t, x_t, model):
    # Kernel_OGD: Online Gradient Descent (OGD) algorithms
    #--------------------------------------------------------------------------
    # Reference:
    # - Martin Zinkevich. Online convex programming and generalized infinitesimal 
    # gradient ascent. In ICML, pages 928?36, 2003.
    #--------------------------------------------------------------------------
    # INPUT:
    #      y_t:     class label of t-th instance;
    #      x_t:     t-th training data instance, e.g., X(t,:);
    #    model:     classifier
    #
    # OUTPUT:
    #    model:     a struct of the weight vector (w) and the SV indexes
    #  hat_y_t:     predicted class label
    #      l_t:     suffered loss

    # Initialization
    loss_type   = model.loss_type     # type of loss
    eta         = model.C             # learning rate

    kernel      = model.kernel        # Kernel method to use
    max_sv      = model.max_sv        # Predefined budget
    alpha       = model.alpha         # Weight vector {-1, 1} per SV
    SV          = model.SV            # active support vectors
    sv_num      = model.sv_num        # Number of support vectors added
    sigma       = model.sigma         # Hyperparameter of Gaussian Kernel
    index       = model.index         # Index for budget maintenance
    
    # Prediction
    last_idx = min(sv_num, max_sv)

    # Similarity vector between current x value and the support vectors
    similarity = np.zeros(1)
    
    if sv_num != 0:
        similarity = kernel(SV, x_t, sigma, last_idx)
        f_t = np.dot(alpha[0:last_idx] ,similarity)
    
    # Make prediction value = 0 when no SVs
    else:
        f_t = 0

    if (f_t>=0):
        hat_y_t = 1
    else:
        hat_y_t = -1

    # Making Update
    eta_t   = eta/np.sqrt(model.t)              # learning rate = eta*(1/sqrt(t)) this learning rate decays over time

    # 0 - 1 Loss
    if loss_type == 0:
        l_t = (hat_y_t != y_t)          # 0 - correct prediction, 1 - incorrect

    # Hinge Loss
    elif loss_type == 1:
        l_t = max(0,1-y_t*f_t) 
        
    # Logistic Loss
    elif loss_type == 2:
        l_t = log(1+exp(-y_t*f_t)) 
    
    # Square Loss
    elif loss_type == 3:
        l_t = 0.5*(y_t - f_t)**2  
    
    else:
        print('Invalid loss type.')
    
    if(l_t > 0):
        SV[index]     = x_t                # Add new SV
        alpha[index]  = y_t                # Update alpha weight for this SV
        index = (index+1) % max_sv

        # Update weight vector with gradient information
        gradient = np.zeros(alpha.shape)
        gradient[0:last_idx] = similarity
        alpha -= eta_t*gradient
    
    model.index   = index
    model.sv_num += 1
    model.SV      = SV
    model.alhpa   = alpha
    model.t = model.t + 1 # iteration counter
    return (model, hat_y_t, l_t)
