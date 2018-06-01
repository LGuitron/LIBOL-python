import numpy as np
def arg_check(task_type, y):
# arg_check: check arguments
#--------------------------------------------------------------------------
# INPUT:
#  task_type: 
#          y: the label vector, e.g., Y(t) is the label of t-th instance;
#
# OUTPUT:
#          0: argument check passed
#          1: argument check failed
#--------------------------------------------------------------------------
# @LIBOL 2012 Contact: chhoi@ntu.edu.sg
#--------------------------------------------------------------------------

    clsnum = len(np.unique(y))
    if ((task_type== 'bc' and clsnum == 2) or (task_type== 'mc' and clsnum >  2)):
        ret = 0
    else:
        ret = 1
    return ret
