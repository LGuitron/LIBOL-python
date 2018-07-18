import numpy as np
from sklearn import preprocessing as pp

# Compute the gaussian kernel between two features
def gaussian_kernel(sv,x, sigma, last_idx):
    diff      = sv[0:last_idx] - x
    magnitude = np.sum(diff**2, axis=1) 
    similarity = np.exp((-magnitude/(2*sigma**2)))
    return similarity



# Polynomial kernel with degree specified by the user
class Polynomial_Kernel:
    
    def __init__(self, degree):
        self.poly = pp.PolynomialFeatures(degree)
    
    def compute(self, x):
        return self.poly.fit_transform(x)
