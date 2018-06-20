import numpy as np
def M_PerceptronM(y_t, x_t, model):
    # M_PerceptronM: Multi-class Perceptron Algorithms with Max-score update.
    # --------------------------------------------------------------------------
    # Reference:
    # Koby Crammer and Yoram Singer. Ultraconservative online algorithms for multiclass problems.
    # Journal of Machine Learning Research, 3:951每991, 2003.
    # --------------------------------------------------------------------------
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
    W           = model.W
    bias        = model.bias
    regularizer = model.regularizer
    
    # Add bias term in feature vector
    if(bias):
        x_t = np.concatenate(([1],x_t))


    # Prediction
    F_t = np.matmul(W,x_t)
    Fmax   = np.max(F_t)
    hat_y_t = np.argmax(F_t) 

    # Making update
    l_t = hat_y_t != y_t             # 0 - correct prediction, 1 - incorrect
    if (l_t > 0):
        
        # compute the hinge loss and support vector
        F_t[int(y_t)]  = -np.inf
        s_t       = np.argmax(F_t) 
        
        model.W[int(y_t),:] = W[int(y_t),:] + x_t
        model.W[s_t,:] = W[s_t,:] - x_t
    
    if(regularizer is not None):
        model.W = regularizer.regularize(model.W)    

    return (model, hat_y_t, l_t)
