import numpy as np
def SCW(y_t, x_t, model):
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
    w     = model.w
    Sigma = model.Sigma
    phi   = model.phi
    C     = model.C
    psi   = 1+(phi**2)/2
    xi    = 1+phi**2

    # Reshape x_t to matrix
    x_t = np.reshape(x_t, (-1,1))

    # Prediction
    f_t = np.matmul(w,x_t)
    if (f_t>=0):
        hat_y_t = 1
    else:
        hat_y_t = -1

    # Making Update
    v_t = np.matmul(np.matmul(x_t.T,Sigma), x_t)  # confidence
    m_t = y_t*f_t;                                # margin
    l_t = phi*np.sqrt(v_t)-m_t;                   # loss
    if l_t > 0:
        alpha_t = max(0,(-1*m_t*psi+np.sqrt((m_t**2*phi**4)/4+v_t*phi**2*xi))/(v_t*xi))
        alpha_t = min(alpha_t, C)
        u_t     = 0.25*(-1*alpha_t*v_t*phi+np.sqrt(alpha_t**2*v_t**2*phi**2+4*v_t))**2
        beta_t  = alpha_t*phi/(np.sqrt(u_t)+alpha_t*phi*v_t);
        S_x_t   = np.matmul(x_t.T,Sigma.T)
        w       = w + alpha_t*y_t*S_x_t;
        Sigma   = Sigma - beta_t*np.matmul(S_x_t.T, S_x_t)
        
    model.w     = w
    model.Sigma = Sigma
    return (model, hat_y_t, l_t)
