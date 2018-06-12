import numpy as np
def Perceptron(y_t, x_t, model):

    # Perceptron: a classical online learning algorithm 
    # --------------------------------------------------------------------------
    # Reference:
    # F. Rosenblatt. The perceptron: A probabilistic model for information
    # storage and organization in the brain.Psychological Review,65:386?07,1958.
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
    w = model.w
    bias  = model.bias
    
    # Add bias term in feature vector
    if(bias):
        x_t = np.concatenate(([1],x_t))
    
    # Prediction
    f_t = np.dot(w,x_t)
    if (f_t>=0):
        hat_y_t = 1
    else:
        hat_y_t = -1
        
    # Loss
    l_t = hat_y_t != y_t
    
    # Update on wrong predictions
    if(l_t):
        model.w = w + y_t*x_t;
    return (model, hat_y_t, l_t)
