import numpy as np
def M_OGD(y_t, x_t, model):
    # M_OGD: Multiclass Online Gradient Descent algorithms (M-OGD)
    #--------------------------------------------------------------------------
    # Reference:
    # - M. Zinkevich. Online convex programming and generalized infinitesimal 
    #   gradient ascent. In ICML 2003.
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
    eta         = model.C;         # learning rate
    bias        = model.bias
    regularizer = model.regularizer
    degree      = model.p_kernel_degree     # Polynomial kernel degree

    # Transform input vector
    if(degree > 1):
        poly = model.poly
        x_t = np.reshape(x_t, (1,-1))       # Reshape x_t to matrix
        x_t  = poly.fit_transform(x_t).T      # Polynomial feature mapping for x_t
        
    # Add bias term in feature vector
    elif(bias):
        x_t = np.concatenate(([1],x_t))
    
    #--------------------------------------------------------------------------
    # Prediction
    #--------------------------------------------------------------------------
    F_t     = np.matmul(W,x_t)
    Fmax    = np.max(F_t)
    hat_y_t = np.argmax(F_t) 

    # compute the hingh loss and support vector
    Fs = np.copy(F_t);
    Fs[int(y_t)] = -np.inf;
    s_t = np.argmax(Fs)
    l_t = np.maximum(0, 1 - (F_t[int(y_t)] - Fs[int(s_t)]))
    
    #--------------------------------------------------------------------------
    # Making Update
    #--------------------------------------------------------------------------
    if (l_t > 0):
        eta_t   = eta/np.sqrt(model.t)
        if degree > 1:
            model.W[int(y_t),:] = W[int(y_t),:] + eta_t*x_t[:,0]
            model.W[int(s_t),:] = W[int(s_t),:] - eta_t*x_t[:,0]
        else:
            model.W[int(y_t),:] = W[int(y_t),:] + eta_t*x_t
            model.W[int(s_t),:] = W[int(s_t),:] - eta_t*x_t
    model.t = model.t + 1 # iteration no

    if(regularizer is not None):
        model.W = regularizer.regularize(model.W)   

    return model, hat_y_t, l_t
    
