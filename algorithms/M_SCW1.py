import numpy as np
def M_SCW1(y_t, x_t, model):
    # M_SCW1: Multiclass Soft Confidence Weight Learning algorithm (M-SCW-I)
    #--------------------------------------------------------------------------
    # Reference:
    # - Soft Confidence-Weighted Learning
    #   Jialei Wang, Peilin Zhao, Steven C.H. Hoi
    #   Nanyang Technological University, Technical Report, 2013
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
    W   = model.W
    C   = model.C
    Sigma = model.Sigma
    phi   = model.phi
    psi   = 1+(phi**2)/2
    xi    = 1+phi**2
    bias  = model.bias
    
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
    Fs = np.copy(F_t);
    Fs[int(y_t)] = -np.inf;
    s_t = np.argmax(Fs)
    m_t = F_t[int(y_t)] - F_t[s_t]
    v_t = np.matmul(np.matmul(x_t,Sigma), x_t.T)  # confidence
    l_t = phi*np.sqrt(v_t)-m_t                    # loss

    #--------------------------------------------------------------------------
    # Making Update
    #--------------------------------------------------------------------------
    if m_t < phi*np.sqrt(v_t):
        alpha_t=max(0, (-m_t*psi+np.sqrt(m_t**2*psi**2-m_t**2*psi+2*v_t*phi**2*psi))/(2*v_t*psi))
        alpha_t = min(alpha_t, C)
        u_t=(1/8)*(-alpha_t*v_t*phi+np.sqrt(alpha_t**2*v_t**2*phi**2+8*v_t))**2
        beta_t=alpha_t*phi/(np.sqrt(2*u_t)+alpha_t*phi*v_t)
        model.W[int(y_t),:] = W[int(y_t),:] + (alpha_t*np.matmul(Sigma, x_t.T)).T
        model.W[s_t,:]      = W[s_t,:] - (alpha_t*np.matmul(Sigma, x_t.T)).T
        model.Sigma         = Sigma - beta_t*np.matmul(np.matmul(Sigma,x_t.T),np.matmul(x_t, Sigma))
        
    return (model, hat_y_t, l_t)
