def CW(y_t, x_t, model):
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
    w     = model.w;
    Sigma = model.Sigma;
    phi   = model.phi;
    psi   = 1+(phi**2)/2;
    xi    = 1+phi**2;

    # Prediction
    f_t = np.dot(w,x_t)
    if (f_t>=0):
        hat_y_t = 1
    else:
        hat_y_t = -1
        

    # Making Update
    v_t = np.dot(np.dot(x_t,Sigma), x_t.T)  # confidence
    m_t = y_t*f_t;                          # margin
    l_t = phi*np.sqrt(v_t)-m_t;             # loss
    if l_t > 0:
        alpha_t = max(0,(-m_t*psi+np.sqrt((m_t**2*phi**4)/4+v_t*phi**2*xi))/(v_t*xi));
        u_t     = 0.25*(-alpha_t*v_t*phi+sqrt(alpha_t**2*v_t**2*phi**2+4*v_t))**2;
        beta_t  = alpha_t*phi/(sqrt(u_t)+alpha_t*phi*v_t);
        S_x_t   = np.dot(x_t,Sigma.T)
        w       = w + alpha_t*y_t*S_x_t;
        Sigma   = Sigma - beta_t*S_x_t'*S_x_t;    
        
    #    w       = w + (alpha_t*y_t*Sigma*x_t')';
    #    Sigma   = Sigma - beta_t*Sigma*x_t'*x_t*Sigma;
    end
    model.w     = w;
    model.Sigma = Sigma;
    %THE END
    return (model, hat_y_t, l_t)
