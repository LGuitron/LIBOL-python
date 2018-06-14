from init_model import Model
import numpy as np
import imp
import time

def ol_train(Y, X, options):

    # ol_train: the main interface to call an online algorithm for training
    #--------------------------------------------------------------------------
    # INPUT:
    #        Y:    the label vector, e.g., Y(t) is the label of t-th instance;
    #        X:    training data matrix, e.g., X(t,:) denotes for t-th instance;
    #  options:    a struct of predefined parameters and training settings;
    # id_list -    a (rand) permutation of the input sequence index: 1,2,...,T;
    #
    # OUTPUT:
    #       model: a struct of the weight vector (w) and the SV indexes
    #      result: a struct of storing training results
    #  err_count - total number of training errors
    #   run_time - cumulative time cost consumed by the algorithm at a tick
    #   mistakes - a vector recording the sequence of online mistake rates
    #      nb_SV - a vector recording the sequence of the SV sizes
    #      ticks - a vector recording the online sequence of time ticks

    #--------------------------------------------------------------------------
    # Initialize parameters
    #--------------------------------------------------------------------------
    ID       = options.id_list  
    n        = len(ID)           # sample size
    d        = X.shape[1]
    t_tick   = options.t_tick
    mistakes = []
    SV       = []
    nb_SV    = []
    ticks    = []
    err_count= 0
    if (options.task_type == 'bc'):
        nb_class=2;
    elif (options.task_type == 'mc'):
        nb_class = len(np.unique(Y))

    model    = Model(options, d, nb_class)   # init model.w=(0,...,0);

    #-------------------------------------------------------------------------
    # BEGIN of the main algorithm
    #--------------------------------------------------------------------------
    start_time = time.time()
    
    
    
    f_ol = options.method                      # get the name of OL function

    # Load python module
    module = imp.load_source( f_ol, './algorithms/' + f_ol +'.py')
    

    func = getattr( module, f_ol )

    for t in range(len(ID)):
        _id  = int(ID[t])
        y_t = Y[_id];
        x_t = X[_id];
        
        #Making prediction & update
        model, hat_y_t, l_t = func(y_t, x_t, model)
    
        # Counting Error
        if (hat_y_t != y_t):
            err_count = err_count + 1; 

        # Add new SV
        if (l_t > 0):
            SV.append(_id)

        # Recording Status
        run_time = time.time() - start_time
        
        if (t % t_tick==0):
            mistakes.append(err_count/(t+1))
            nb_SV.append(len(SV))
            ticks.append(run_time)

    #--------------------------------------------------------------------------
    # END OF the main algorithm and OUTPUT
    #--------------------------------------------------------------------------
    run_time = time.time() - start_time
    result = (run_time, err_count, mistakes, ticks, nb_SV)
    model.SV = SV;
    print(options.method, 'The cumulative mistake rate = ' , round(100*err_count/n, 3), '% (' , err_count , '/', n , '), CPU time cost: ' , round(run_time,3), "s")

    return model, result
