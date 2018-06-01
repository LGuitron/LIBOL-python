from libsvm import libsvm
import numpy as np

def load_data(dataset_name, file_format, task_type):
    
    #case 'mat' % matlab format
    #        load(sprintf('data/%s.mat',dataset_name));
    #        [n,d]   = size(data);
    #        xt      = data(:,2:d); y = data(:,1); 

    #case 'arff'
    #data    = arff2matlab(sprintf('data/%s.arff',dataset_name));
    #[n,d]   = size(data);
    #xt      = data(:,1:d-1); y = data(:,d);
    #libc = cdll.msvcrt
    
    if file_format == 'libsvm':
        y, xt, n = libsvm.svm_read_problem(dataset_name)
        #y, xt, n = libsvm.svm_read_problem('data/' + dataset_name)
    
    else:
        print('The file format is not supported.') 
        return None

    #for binary-class data set in binary-class settings
    if (np.unique(y).shape[0] and task_type=='bc'):
        y = y - min(y)
        y = y/max(y)
        y = 2*y - 1
    
    #print(y)
    # for binary-class data set in multi-class setting
    #if ((length(unique(y))==2) && strcmp(task_type,'mc') && (min(y)<=0)),
    #y = y - min(y) + 1; % shift from -1,0,1,2... to 1,2,3,4,...
    
    # for multi-class data set
    #if ((length(unique(y))>2) && (min(y)<=0)),
    #y = y - min(y) + 1; % shift from -1,0,1,2... to 1,2,3,4,...

    return (xt, y, n)
