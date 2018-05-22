import numpy as np
def PA2(y_t, x_t, model):
    # PA: Passive-Aggressive (PA) learning algorithms
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
    w     = model.w
    C     = model.C

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
        gamma_t = l_t/(s_t+(1/(2*C)))      # PA-II
        model.w = w + gamma_t*y_t*x_t
    return (model, hat_y_t, l_t)
