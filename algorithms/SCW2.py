import numpy as np
def SCW2(y_t, x_t, model):
    # SCW-II: Soft Confidence-Weighted Learning Algorithm (variant 2)
    #--------------------------------------------------------------------------
    # Reference:
    # "Exact Soft Confidence-Weighted Learning", Jielei Wang, Peilin Zhao,
    # Steven C.H. Hoi, ICML2012, 2012.
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
    phi         = model.phi
    C           = model.C
    psi         = 1+(phi**2)/2
    xi          = 1+phi**2
    bias        = model.bias
    regularizer = model.regularizer
    
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
    v_t = np.matmul(np.matmul(x_t,Sigma), x_t.T)  # confidence
    m_t = y_t*f_t;                                # margin
    n_t = v_t + 1/(2*C)
    l_t = phi*np.sqrt(v_t)-m_t;                   # loss
    if l_t > 0:
        alpha_t = max(0,(-(2*m_t*n_t+phi**2*m_t*v_t) + np.sqrt(phi**4*m_t**2*v_t*2+4*n_t*v_t*phi**2*(n_t+v_t*phi*2)))/(2*(n_t**2+n_t*v_t*phi**2)))
        u_t     = 0.25*(-1*alpha_t*v_t*phi+np.sqrt(alpha_t**2*v_t**2*phi**2+4*v_t))**2
        beta_t  = alpha_t*phi/(np.sqrt(u_t)+alpha_t*phi*v_t);
        S_x_t   = np.matmul(x_t,Sigma.T)
        w       = w + alpha_t*y_t*S_x_t;
        Sigma   = Sigma - beta_t*np.matmul(S_x_t.T, S_x_t)
        
    model.w     = w
    model.Sigma = Sigma
    
    
    if(regularizer is not None):
        model.w = regularizer.regularize(model.w)
    
    
    return (model, hat_y_t, l_t)
