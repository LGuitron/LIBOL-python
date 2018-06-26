import numpy as np
def SOP(y_t, x_t, model):
    # SOP: Second Order Perceptron algorithm
    #--------------------------------------------------------------------------
    # Reference:
    # N. Cesa-Bianchi, A. Conconi, and C. Gentile, A second-order Perceptron
    # algorithm, SIAM Journal on Computing, 34(3):640-668. SIAM, 2005.
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
    bias        = model.bias

    # Reshape x_t to matrix
    x_t = np.reshape(x_t, (1,-1))
    
    # Add bias term in feature vector
    if(bias):
        x_t = np.concatenate(([[1]],x_t), axis = 1)
        
    # Prediction
    S_x_t   = np.matmul(x_t, Sigma.T)
    v_t = np.matmul(x_t, S_x_t.T)
    
    
    beta_t  = 1/(v_t+1);
    Sigma_t   = Sigma - beta_t*np.matmul(S_x_t.T, S_x_t)
    
    f_t     = np.matmul(np.matmul(w,Sigma), x_t.T)
    if (f_t>=0):
        hat_y_t = 1
    else:
        hat_y_t = -1

    # Making Update
    l_t = (hat_y_t != y_t) # 0 - correct prediction, 1 - incorrect    
    if (l_t > 0):    
        w     +=  y_t*x_t    
    model.w = w
    model.Sigma = Sigma_t

    return (model, hat_y_t, l_t)
