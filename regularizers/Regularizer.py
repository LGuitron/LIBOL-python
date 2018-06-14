import numpy as np

class L0:
    # L0 regularization (Rounding of small coefficients to 0)
    # Parameters:
    #          theta:     Threshold absolute value to decrease coefficients
    #    update_freq:     Frequency of coefficient rounding
    
    def __init__(self, theta, update_freq):
        self.c_step      = 0
        self.theta       = theta
        self.update_freq = update_freq
        

    def regularize(self, w):
        if(self.c_step == self.update_freq):
            self.c_step = 0
            w[np.where(np.absolute(w) < self.theta)] = 0
        else:
            self.c_step += 1
        return w


class TGD:
    # Truncated Gradient Descent
    # Parameters:
    #          theta:     Threshold absolute value to decrease coefficients
    #              g:     Rounding aggressivenes
    #    update_freq:     Frequency of coefficient rounding
    
    def __init__(self, theta, g, update_freq):
        self.c_step      = 0
        self.theta       = theta
        self.g           = g
        self.update_freq = update_freq

    # Learning rate affects rounding aggressivenes
    def regularize(self, w, learning_rate = 1):

        alpha = self.g * learning_rate

        if(self.c_step == self.update_freq):
            self.c_step = 0

            # Decrease values between 0 and theta by alhpa (Positive weights)
            interval    = np.where((0 < w) & (w < self.theta))
            zero_vector = np.zeros(w[interval].shape) 
            w[interval] = np.maximum(zero_vector, w[interval]-alpha)
            
            # Increase values between -theta and 0 by alpha (Negative weights)
            interval    = np.where((-self.theta < w) & (w < 0))
            zero_vector = np.zeros(w[interval].shape) 
            w[interval] = np.minimum(zero_vector, w[interval]+alpha)
        else:
            self.c_step += 1
        return w
