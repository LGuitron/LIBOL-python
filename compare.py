import argparse
from load_data import load_data
from arg_check import arg_check
from handle_parameters import handle_parameters
from run import run
from plot import plot

def compare(task_type, dataset_name, file_format, nb_runs, print_results = True, showPlot = True):
    #--------------------------------------------------------------------------
    # This function runs all of the algorithms available for either bc or mc
    # Algorithms run with the options specified in the init_options.py file
    #Examples:
    #   run_experiment_bc('svmguide3','mat','m')
    #   run_experiment_bc('w1a','libsvm','c')
    #   run_experiment_bc('sonar','arff','m')

    if task_type == 'bc':
        algorithms = ['Perceptron','Gaussian_Kernel_Perceptron','PA','PA1','PA2','OGD','Gaussian_Kernel_OGD','SOP','CW','SCW','SCW2','AROW','NAROW']
    
    elif task_type == 'mc':
        algorithms = ['M_PerceptronM','M_PerceptronS','M_PerceptronU','M_PA','M_PA1','M_PA2','M_OGD','M_CW','M_SCW1','M_SCW2','M_AROW']
        
    else: 
        print("Unknown task type")
        return

    # Run all algorithms of this type
    run_stats = []
    
    for algorithm_name in algorithms:
        stats = run(task_type, algorithm_name, dataset_name, file_format, nb_runs = nb_runs, print_results = print_results)
        run_stats.append(stats)
    
    if(showPlot):
        plot(algorithms, run_stats, dataset_name)
        
    return run_stats

if __name__ == '__main__':
    task_type, _, dataset_name, file_format, nb_runs = handle_parameters()
    compare(task_type, dataset_name, file_format, nb_runs)
