import numpy as np
def PA(y_t, x_t, model):
    # PA: Passive-Aggressive (PA) learning algorithms
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

    # Initialization
    w           = model.w
    bias        = model.bias
    degree      = model.p_kernel_degree     # Polynomial kernel degree

    # Transform input vector
    if(degree > 1):
        poly = model.poly
        x_t = np.reshape(x_t, (1,-1))       # Reshape x_t to matrix
        x_t  = poly.fit_transform(x_t).T      # Polynomial feature mapping for x_t

    # Add bias term in feature vector
    elif(bias):
        x_t = np.concatenate(([1],x_t))


    # Prediction
    f_t = np.dot(w,x_t)
    
    if (f_t>=0):
        hat_y_t = 1
    else:
        hat_y_t = -1

    # Hinge Loss
    l_t = max(0,1-y_t*f_t)
    
    # Update on non-zero loss
    if (l_t > 0):
        s_t = np.linalg.norm(x_t)**2
        if (s_t > 0):
            gamma_t = l_t/s_t            # gamma_t = min(C,l_t/s_t);(PA-I)
        else:
            gamma_t = 1                  # special case when all x goes zero.
            
        if(degree > 1):
            model.w = w + (gamma_t*y_t*x_t).T
        else:
            model.w = w + gamma_t*y_t*x_t

    return (model, hat_y_t, l_t)
