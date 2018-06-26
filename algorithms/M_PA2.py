import numpy as np
def M_PA2(y_t, x_t, model):
    # M_PA: Multiclass Passive-Aggressive (M-PA) learning algorithms
    #--------------------------------------------------------------------------
    # Reference:
    # - Koby Crammer, Ofer Dekel, Joseph Keshet, Shai Shalev-Shwartz, and Yoram
    # Singer. Online passive-aggressive algorithms. JMLR, 7:551?85, 2006.
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
    W           = model.W
    C           = model.C
    bias        = model.bias

    # Add bias term in feature vector
    if(bias):
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
        eta_t = l_t/(2*np.linalg.norm(x_t)**2+1/(2*C))
        model.W[int(y_t),:] = W[int(y_t),:] + eta_t*x_t
        model.W[int(s_t),:] = W[int(s_t),:] - eta_t*x_t           

    return model, hat_y_t, l_t
