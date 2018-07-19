from libsvm import libsvm
import numpy as np

def load_data(dataset_name, file_format, task_type):
    
    if file_format == 'libsvm':
        y, xt, n = libsvm.svm_read_problem(dataset_name)
    
    else:
        print('The file format is not supported.') 
        return None

    #for binary-class data set in binary-class settings
    if (np.unique(y).shape[0] and task_type=='bc'):
        y = y - min(y)
        y = y/max(y)
        y = 2*y - 1
        
    # for binary-class data set in multi-class setting
    if (len(np.unique(y)) == 2 and task_type == 'mc' and min(y)<=0):
        y = y - min(y)  # shift from -1,0,1,2... to 0,1,2,3,...

    # for multi-class data set 
    if (len(np.unique(y))>2 and min(y)!=0):
        y = y - min(y)  # shift from -1,0,1,2... to 0,1,2,3,...

    return (xt, y, n)
