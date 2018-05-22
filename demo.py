import sys
import argparse
import numpy as np
from load_data import load_data
from init_options import Options
from ol_train import ol_train

def demo():
    
    # Parser with default options if no arguments are given
    parser = argparse.ArgumentParser(description='Online learning algorithms selection')
    parser.add_argument('-t', type=str, default='bc')            # Classification Task
    parser.add_argument('-a', type=str, default='Perceptron')    # OL algorithm
    parser.add_argument('-d', type=str, default='a7a.t')         # dataset
    parser.add_argument('-f', type=str, default='libsvm')        # file format
    args = parser.parse_args()

    task_type = args.t
    algorithm_name = args.a
    dataset_name = args.d
    file_format = args.f

    return_vals = load_data(dataset_name, file_format, task_type) 
    
    if return_vals is None:
        sys.exit("Argument error.")
    
    xt, y, n = return_vals

    #check argument
    #if arg_check(task_type, y) ~= 0
    #  disp(['Error: Dataset is not for "' task_type '" task.']);
    #  return;
    #end

    #initializing paramters
    _options = Options(algorithm_name, n, task_type); 

    # START selecting paramters...
    #options = CV_algorithm(y, xt, options); % auto parameter selection
    # end of paramter selection.

    # START generating test ID sequence...
    nb_runs = 20
    ID_list = np.zeros((nb_runs,n))
    

    
    for i in range (nb_runs):
        ID_list[i]= np.random.permutation(n)


    # Arrays with algorithm stats
    err_count_arr = np.zeros(nb_runs)
    nSV_arr = np.zeros(nb_runs)
    time_arr = np.zeros(nb_runs)

    for i in range(nb_runs):
        print('running on the', i , '-th trial...')
        _options.id_list = ID_list[i]
        model, result = ol_train(y, xt, _options)

        # Stats for this run
        run_time, err_count, mistakes, ticks, nb_SV = result
        err_count_arr[i] = err_count
        nSV_arr[i]       = len(model.SV)
        time_arr[i]      = run_time

    print('-------------------------------------------------------------------------------')
    print('Dataset name: ',dataset_name, '( n=' , n, ' d=' ,xt.shape[1],  ' m=' , len(np.unique(y)), ') \t nb of runs (permutations): ',nb_runs)
    print('-------------------------------------------------------------------------------')
    print('Algorithm: ', algorithm_name)
    print("mistake rate: " , round(np.mean(err_count_arr)/n, 3), "+/-", round(np.std(err_count_arr)/n,3))
    print("nb of updates: " , round(np.mean(nSV_arr), 3), "+/-", round(np.std(nSV_arr),3))
    print("cpu time (seconds): " , round(np.mean(time_arr),3) , "+/-", round(np.std(time_arr),3))
    print('-------------------------------------------------------------------------------')

demo()
