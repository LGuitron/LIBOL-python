import numpy as np
from ol_train import ol_train
def CV_algorithm(Y,X,options, print_trials):
# CV_algorithm: This aims to choose best paramters via validation automatically.
#--------------------------------------------------------------------------
#  INPUT:
#        Y:    the label vector, e.g., Y(t) is the label of t-th instance;
#        X:    training data, e.g., X(t,:) denotes for t-th instance;
#  options:    a struct of predefined parameters
# OUTPUT:
#  options:    a struct of parameters with best validated values
#--------------------------------------------------------------------------
# @LIBOL 2012 Contact: chhoi@ntu.edu.sg
#--------------------------------------------------------------------------
    
    method = options.method
    method = method.upper()
    if(method=='PERCEPTRON' or method=='ROMMA' or method=='AROMMA' or method=='PA'):
        pass

    elif(method=='PA1' or method=='PA2' or method=='NHERD' or method=='WINNOW' or method=='OGD'):
        options = best_parameter_C(Y,X,options, print_trials)
        
    
    elif(method=='ALMA'):
        options = best_parameter_C(Y,X,options, print_trials)
        options = best_parameter_eta(Y,X,options, print_trials)   # i.e., alpha
        options = best_parameter_p(Y,X,options, print_trials)     # i.e., p = [2,4,6,8,10]                
    
    elif(method=='CW'):
        options = best_parameter_eta(Y,X,options, print_trials);
    
    elif(method=='AROW'):
        options = best_parameter_C(Y,X,options, print_trials)
    
    elif(method=='SOP'):
        options.SOP_a = 1;

    elif(method=='IELLIP'):
        options = best_parameter_b(Y,X,options, print_trials)
    
    elif(method=='SCW'or method=='SCW2'):
        options = best_parameter_C(Y,X,options, print_trials);
        options = best_parameter_eta(Y,X,options, print_trials);

    elif(method=='NAROW'):
        options = best_parameter_C(Y,X,options, print_trials)

    elif(method=='M_ROMMA' or method=='M_AROMMA'):
        pass
    
    elif(method=='M_PERCEPTRONM'or method=='M_PERCEPTRONU'or method=='M_PERCEPTRONS'):
        pass

    elif(method=='M_PA'or method=='M_PA1'or method=='M_PA2'or method=='M_OGD'):
        options = best_parameter_C(Y,X,options, print_trials)
        
    elif (method=='M_CW'):
        options = best_parameter_eta(Y,X,options, print_trials)

    elif (method=='M_SCW'or method=='M_SCW2'):
        options = best_parameter_C(Y,X,options, print_trials)
        options = best_parameter_eta(Y,X,options, print_trials)

    elif (method=='M_AROW'):
        options = best_parameter_C(Y,X,options, print_trials)

    elif (method=='NEW_ALGORITHM'):
        # find the best paramters via validation below
        # options = best_paramter_....
        pass

    else:
        print('Unknown method.');
    return options

    

def best_parameter_C(Y,X,options, print_trials):
    best_err_count  = X.shape[0]
    value_domain = 2**np.arange(-4.0,4.0,1.0)

    for i in range(len(value_domain)):
        options.C       = value_domain[i]
        if print_trials:
            print('CV: validating parameter C = ',options.C)
        [model,result]  = ol_train(Y,X,options, print_trials)
        err_count = result[1]
        if err_count <= best_err_count:
            best_err_count = err_count
            best_value = value_domain[i]


    options.C = best_value
    if print_trials:
        print('CV_result: The best value of parameter C =',options.C)
    return options

def best_parameter_eta(Y,X,options, print_trials):
    best_err_count  = X.shape[0]
    value_domain = np.arange(0.55,0.95,0.05)
    
    for i in range(len(value_domain)):
        options.eta       = value_domain[i]
        if print_trials:
            print('CV: validating parameter eta = ',options.eta)
        [model,result]  = ol_train(Y,X,options, print_trials)
        err_count = result[1]
        if err_count <= best_err_count:
            best_err_count = err_count
            best_value = value_domain[i]

    options.eta = best_value;
    if print_trials:
        print('CV_result: The best value of parameter eta =',options.eta)
    return options

def best_parameter_b(Y,X,options, print_trials):
    best_err_count  = X.shape[0]
    value_domain = np.arange(0.1,0.9,0.1)
    
    for i in range(len(value_domain)):
        options.b       = value_domain[i]
        if print_trials:
            print('CV: validating parameter b = ',options.b)
        [model,result]  = ol_train(Y,X,options, print_trials)
        err_count = result[1]
        if err_count <= best_err_count:
            best_err_count = err_count
            best_value = value_domain[i]

    options.b = best_value;
    if print_trials:
        print('CV_result: The best value of parameter b =',options.b)
    return options

def best_parameter_p(Y,X,options, print_trials):
    best_err_count  = X.shape[0]
    value_domain = np.arange(2,10,2)
    
    for i in range(len(value_domain)):
        options.p       = value_domain[i]
        if print_trials:
            print('CV: validating parameter p = ',options.p)
        [model,result]  = ol_train(Y,X,options, print_trials)
        err_count = result[1]
        if err_count <= best_err_count:
            best_err_count = err_count
            best_value = value_domain[i]

    options.p = best_value;
    if print_trials:
        print('CV_result: The best value of parameter p =',options.eta)
    return options
