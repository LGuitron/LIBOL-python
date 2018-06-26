import numpy as np
def AROW(y_t, x_t, model):
    # CW: Confidence Weighted Online Leanring Algorithm 
    #--------------------------------------------------------------------------
    # Reference:
    # Exact Convex Confidence-Weighted Learning, Koby Crammer, Mark Dredze and
    # Fernando Pereira, NIPS, 2008.
    #--------------------------------------------------------------------------
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
    r           = model.r
    bias        = model.bias
    
    # Reshape x_t to matrix
    x_t = np.reshape(x_t, (1,-1))
    
    # Add bias term in feature vector
    if(bias):
        x_t = np.concatenate(([[1]],x_t), axis = 1)
    
    # Prediction
    f_t = np.matmul(w,x_t.T)
    if (f_t>=0):
        hat_y_t = 1
    else:
        hat_y_t = -1

    # Making Update
    v_t = np.matmul(np.matmul(x_t,Sigma), x_t.T)    # confidence
    m_t = f_t;                                      # margin                 
    l_t = max(0,1-m_t*y_t)                          # hinge loss
    if l_t > 0:
        beta_t  = 1/(v_t + r)
        alpha_t = l_t*beta_t
        S_x_t   = np.matmul(x_t,Sigma.T)
        w       = w + alpha_t*y_t*S_x_t
        Sigma   = Sigma - beta_t*np.matmul(S_x_t.T, S_x_t)
        
    model.w     = w
    model.Sigma = Sigma
    
    return (model, hat_y_t, l_t)
