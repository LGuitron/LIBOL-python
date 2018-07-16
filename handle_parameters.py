import argparse
def handle_parameters():

    # Parser with default options if no arguments are given
    parser = argparse.ArgumentParser(description='Online learning algorithms selection')
    parser.add_argument('-t', type=str, help='Classification task (Default bc): {bc, mc} ',dest='task',  default='bc')                           # Classification Task
    parser.add_argument('-a', type=str, help='OL algorithm to run (Default Perceptron):{Perceptron, PA, PA1, PA2, OGD,SOP, CW, SCW, SCW2, AROW}', dest='algorithm', default='Perceptron')                                                                                                                        # OL algorithm
    parser.add_argument('-d', type=str, help='Path to dataset (Default ./data/test/bc/breast_cancer_scale.txt)', dest='path_to_dataset', default='./data/test/bc/breast_cancer_scale.txt') # dataset
    
    parser.add_argument('-f', type=str, help='Input file format (Default libsvm)', dest='file_format',default='libsvm')                          # file format
    args = parser.parse_args()

    return args.task, args.algorithm, args.path_to_dataset, args.file_format
