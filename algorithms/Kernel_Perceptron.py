import numpy as np

def Kernel_Perceptron(y_t, x_t, model):

    # KernelPerceptron: Non-linear perceptron algorithm by using the kernel trick
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
    kernel      = model.kernel        # Kernel method to use
    max_sv      = model.max_sv        # Predefined budget
    alpha       = model.alpha         # Weight vector {-1, 1} per SV
    SV          = model.SV            # active support vectors
    sv_num      = model.sv_num        # Number of support vectors added
    sigma       = model.sigma         # Hyperparameter of Gaussian Kernel
    index       = model.index         # Index for budget maintenance
    
    # Prediction
    last_idx = min(sv_num, max_sv)
    if sv_num != 0:
        f_t = np.dot(alpha[0:last_idx] ,kernel(SV, x_t, sigma, last_idx))

    # Make prediction value = 0 when no SVs
    else:
        f_t = 0
    
    if (f_t>=0):
        hat_y_t = 1
    else:
        hat_y_t = -1
        
    # Loss
    l_t = hat_y_t != y_t                  # Hinge Loss

    # Update on wrong predictions
    if(l_t > 0):
        
        SV[index]     = x_t                # Add new SV
        alpha[index]  = y_t                # Update alpha weight for this SV
        index = (index+1) % max_sv
        
        model.index   = index
        model.sv_num += 1
        model.SV      = SV
        model.alhpa   = alpha
    
    return (model, hat_y_t, l_t)
