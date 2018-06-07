import numpy as np
def M_PerceptronU(y_t, x_t, model):
    # M_PerceptronU: Multi-class Perceptron Algorithms with uniform update.
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
    W = model.W

    # Prediction
    F_t = np.matmul(W,x_t)
    Fmax   = np.max(F_t)
    hat_y_t = np.argmax(F_t) 

    E = np.where(F_t > F_t[int(y_t)])[0]
    norm_E = len(E)
    
    # Making update
    l_t = hat_y_t != y_t             # 0 - correct prediction, 1 - incorrect
    if (l_t > 0):
        model.W[int(y_t),:] = W[int(y_t),:] + x_t
        
        if norm_E > 0:
            for i in range(norm_E):
                s_t = E[i]
                model.W[s_t,:] = W[s_t,:] - (1/norm_E)*x_t

    return (model, hat_y_t, l_t)
