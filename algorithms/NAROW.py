import numpy as np
def NAROW(y_t, x_t, model):
    # NAROW: New Adaptive Regularization Of Weights (AROW) algorithm
    #--------------------------------------------------------------------------
    # Reference:
    # Orabona, Francesco and Crammer, Koby. "New adaptive algorithms for online
    # classification." In NIPS, pp. 1840?848, 2010
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
    w           = model.w
    Sigma       = model.Sigma
    b           = model.b
    bias        = model.bias
    degree      = model.p_kernel_degree     # Polynomial kernel degree

    # Reshape x_t to matrix
    x_t = np.reshape(x_t, (1,-1))

    # Transform input vector
    if(degree > 1):
        poly = model.poly
        x_t  = poly.fit_transform(x_t)     # Polynomial feature mapping for x_t

    # Add bias term in feature vector
    elif(bias):
        x_t = np.concatenate(([[1]],x_t), axis = 1)

    # Prediction
    f_t = np.matmul(w,x_t.T)
    if (f_t>=0):
        hat_y_t = 1
    else:
        hat_y_t = -1

    # Making Update
    v_t = np.matmul(np.matmul(x_t,Sigma), x_t.T)    # confidence
    m_t = y_t*f_t;                                  # margin     
    l_t = 1 - m_t;                                  # hinge loss

    if l_t > 0:
        
        chi_t = np.matmul(np.matmul(x_t, Sigma),x_t.T)      # inv(A_{t-1}^{-1})?   
        if chi_t > 1/b:
            r_t = chi_t/(b*chi_t-1)
        else:
            r_t = np.inf

        beta_t  = 1/(v_t + r_t)
        alpha_t = max(0, 1-m_t)*beta_t
        S_x_t   = np.matmul(x_t,Sigma.T)
        w       = w + alpha_t*y_t*S_x_t;
        Sigma   = Sigma - beta_t*np.matmul(S_x_t.T, S_x_t)        
        
    model.w     = w
    model.Sigma = Sigma
        
    return (model, hat_y_t, l_t)
