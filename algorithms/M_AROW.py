import numpy as np
def M_AROW(y_t, x_t, model):
    # M_AROW: Multi-class Adaptive Regularization of Weights
    #--------------------------------------------------------------------------
    # Reference:
    # - Adaptive Regularization of Weight Vectors 
    #   Koby Crammer, Alex Kulesza, Mark Dredze.
    #   Machine Learning, 2013 
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
    #--------------------------------------------------------------------------
    # @LIBOL 2012 Contact: chhoi@ntu.edu.sg
    #--------------------------------------------------------------------------



    #--------------------------------------------------------------------------
    # Initialization
    #--------------------------------------------------------------------------
    W           = model.W;
    Sigma       = model.Sigma;
    r           = model.r;
    bias        = model.bias

    # Reshape x_t to matrix
    x_t = np.reshape(x_t, (1,-1))
    
        # Add bias term in feature vector
    if(bias):
        x_t = np.concatenate(([[1]],x_t), axis = 1)
    
    #--------------------------------------------------------------------------
    # Prediction
    #--------------------------------------------------------------------------
    F_t     = np.matmul(W,x_t.T)
    Fmax    = np.max(F_t)
    hat_y_t = np.argmax(F_t) 

    # compute the hinge loss and support vector
    Fs = np.copy(F_t)
    Fs[int(y_t)] = -np.inf
    s_t = np.argmax(Fs)
    m_t = F_t[int(y_t)] - F_t[s_t]
    v_t = np.matmul(np.matmul(x_t,Sigma), x_t.T)  # confidence
    l_t = max(0,1-m_t) # hinge loss

    #--------------------------------------------------------------------------
    # Making Update
    #--------------------------------------------------------------------------
    if l_t > 0:
        beta_t  = 1/(v_t + r);
        alpha_t = l_t*beta_t;
        model.W[int(y_t),:] = W[int(y_t),:] + (alpha_t*np.matmul(Sigma,x_t.T)).T
        model.W[s_t,:]      = W[s_t,:] - (alpha_t*np.matmul(Sigma,x_t.T)).T
        model.Sigma         = Sigma - beta_t*np.matmul(np.matmul(Sigma,x_t.T),np.matmul(x_t, Sigma)) 

    return (model, hat_y_t, l_t)
