import numpy as np

def svm_read_problem(data_file_name):
    """
    svm_read_problem(data_file_name) -> [y, x]
    Read LIBSVM-format data from data_file_name and return labels y
    and data instances x.
    """
    prob_y = []
    prob_x = []
    max_ind = 0
    n = 0
    for line in open(data_file_name):
        line = line.split(None, 1)
        # In case an instance with all zero features
        if len(line) == 1: line += ['']
        label, features = line
        xi = {}
        for e in features.split():
            ind, val = e.split(":")
            max_ind = max(int(ind), max_ind)
            xi[int(ind)] = float(val)
        prob_y += [float(label)]
        prob_x += [xi]
        n += 1

    y = np.array(prob_y)
    x = np.zeros((n, max_ind))

    # X to numpy array
    for i in range(len(prob_x)):
        for index in prob_x[i]:
            x[i][index-1] = prob_x[i][index]

    return (y, x, n)
