import numpy as np
def OGD(y_t, x_t, model):
    # OGD: Online Gradient Descent (OGD) algorithms
    #--------------------------------------------------------------------------
    # Reference:
    # - Martin Zinkevich. Online convex programming and generalized infinitesimal 
    # gradient ascent. In ICML, pages 928?36, 2003.
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
    loss_type   = model.loss_type     # type of loss
    eta         = model.C             # learning rate
    bias        = model.bias
    regularizer = model.regularizer   # Regularizer method
    
    # Add bias term in feature vector
    if(bias):
        x_t = np.concatenate(([1],x_t))
    
    # Prediction
    f_t = np.dot(w,x_t)
    if (f_t>=0):
        hat_y_t = 1
    else:
        hat_y_t = -1

    # Making Update
    eta_t   = eta/np.sqrt(model.t)              # learning rate = eta*(1/sqrt(t)) this learning rate decays over time

    # 0 - 1 Loss
    if loss_type == 0:
        l_t = (hat_y_t != y_t)          # 0 - correct prediction, 1 - incorrect
        if(l_t > 0):
            w += eta_t*y_t*x_t                                   # Update w with hinge loss derivative

    # Hinge Loss
    elif loss_type == 1:
        l_t = max(0,1-y_t*f_t) 
        if(l_t > 0):
            w += eta_t*y_t*x_t                                   # Update w with hinge loss derivative


    # Logistic Loss
    elif loss_type == 2:
        l_t = log(1+exp(-y_t*f_t)) 
        if(l_t > 0):
            w += eta_t*y_t*x_t*(1/(1+exp(y_t*f_t)))                                  # Update w with hinge loss derivative
            
    # Square Loss
    elif loss_type == 3:
        l_t = 0.5*(y_t - f_t)**2  
        if(l_t > 0):
            w += -eta_t*(f_t-y_t)*x_t                                 # Update w with hinge loss derivative

    else:
        print('Invalid loss type.')
    model.w = w
    if(regularizer is not None):
        model.w = regularizer.regularize(model.w, learning_rate=eta_t)
    model.t = model.t + 1 # iteration counter
    return (model, hat_y_t, l_t)
