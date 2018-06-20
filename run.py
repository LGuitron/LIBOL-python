import argparse
import numpy as np
from math import floor
from load_data import load_data
from init_options import Options
from ol_train import ol_train
from CV_algorithm import CV_algorithm
from arg_check import arg_check
from handle_parameters import handle_parameters

def run(task_type, algorithm_name, dataset_name, file_format, print_trials = True):

    return_vals = load_data(dataset_name, file_format, task_type) 
    
    if return_vals is None:
        sys.exit("Argument error.")
    
    xt, y, n = return_vals

    #check argument
    if arg_check(task_type, y) != 0:
      print("Error: Dataset is not for ", task_type , " task.")
      return

    #initializing paramters
    _options = Options(algorithm_name, n, task_type)

    # START selecting paramters...
    _options = CV_algorithm(y, xt, _options, print_trials)      # auto parameter selection
    # end of paramter selection.

    # START generating test ID sequence...
    nb_runs = 20
    ID_list = np.zeros((nb_runs,n))

    for i in range (nb_runs):
        ID_list[i]= np.random.permutation(n)


    # Arrays with algorithm stats
    err_count_arr   = np.zeros(nb_runs)
    nSV_arr         = np.zeros(nb_runs)
    time_arr        = np.zeros(nb_runs)
    mistakes_arr    = np.zeros((floor(n/_options.t_tick), nb_runs))
    nb_SV_cum_arr   = np.zeros((floor(n/_options.t_tick), nb_runs))
    time_cum_arr    = np.zeros((floor(n/_options.t_tick), nb_runs))

    for i in range(nb_runs):
        if(print_trials):
            print('running on the', i , '-th trial...')

        _options.id_list = ID_list[i]
        model, result = ol_train(y, xt, _options, print_trials)

        # Stats for this run
        run_time, err_count, mistakes, ticks, nb_SV, captured_t = result
        err_count_arr[i]  = err_count
        nSV_arr[i]        = model.final_nb_SV
        time_arr[i]       = run_time
        
        
        
        # Algorithm stats after a specific number of updating steps
        mistakes_arr[:,i]  = mistakes
        nb_SV_cum_arr[:,i] = nb_SV
        time_cum_arr[:,i]  = ticks
        

    mean_error_count  = round(np.mean(err_count_arr)/n, 4)
    mean_update_count = round(np.mean(nSV_arr), 4)
    mean_time         = round(np.mean(time_arr),4)
    mean_mistakes     = np.mean(mistakes_arr, axis = 1)


    print('-------------------------------------------------------------------------------')
    print('Dataset name: ',dataset_name, '( n=' , n, ' d=' ,xt.shape[1],  ' m=' , len(np.unique(y)), ') \t nb of runs (permutations): ',nb_runs)
    print('-------------------------------------------------------------------------------')
    print('Algorithm: ', algorithm_name)
    print("mistake rate: " , mean_error_count, "+/-", round(np.std(err_count_arr)/n,4))
    print("nb of updates: " , mean_update_count, "+/-", round(np.std(nSV_arr),4))
    print("cpu time (seconds): " , mean_time , "+/-", round(np.std(time_arr),4))
    print('-------------------------------------------------------------------------------')
    
    return (mean_error_count, mean_update_count, mean_time, np.mean(mistakes_arr, axis=1), np.mean(nb_SV_cum_arr, axis=1), np.mean(time_cum_arr, axis=1), captured_t)

if __name__ == '__main__':
    task_type, algorithm_name, dataset_name, file_format = handle_parameters()
    run(task_type, algorithm_name, dataset_name, file_format)
