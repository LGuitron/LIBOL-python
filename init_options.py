import numpy as np

class Options:

    def __init__(self, method,n,task_type):
   
    # init_options: initialize the options for each method
    #--------------------------------------------------------------------------
    # INPUT:
    #       method:     method name
    #       n:          number of training instances in the database
    #       task_type:  type of task (bc or mc)

        self.method = method
        self.t_tick = round(n/15)    #10
        self.task_type = task_type
        self.id_list = np.random.permutation(n)
        
        UPmethod = method.upper()
        if ((len(UPmethod) > 2) and (UPmethod[len(UPmethod)-1], 'C')):
            self.language = 1     #C 
        else:
            self.language = 0     # Python

        if (task_type == 'bc'):
            
            if (UPmethod == 'PERCEPTRON' or UPmethod =='ROMMA' or UPmethod == 'AROMMA' or 
                UPmethod =='PERCEPTRON_C' or UPmethod =='PA' or UPmethod =='PA_C' or 
                UPmethod =='ROMMA_C' or  UPmethod =='AROMMA_C'):
                    pass
            
            elif (UPmethod == 'WINNOW' or UPmethod =='PA1' or UPmethod =='PA2' or UPmethod =='PA1_C' or UPmethod =='PA2_C'):
                self.C = 1
            
            elif (UPmethod == 'ALMA' or UPmethod =='ALMA_C'):
                self.eta = 0.9  # alpha(eta) \in (0,1]
                self.p   = 2
                self.C   = sqrt(2)

            elif (UPmethod == 'OGD' or UPmethod =='OGD_C'):
                self.t = 1           # iteration no, learning rate eta_t = 1/sqrt(t)
                self.loss_type = 1   # type of loss (0, 0-1 loss, 1 - hinge, 2-log, 3-square )
                self.C = 1

            elif (UPmethod == 'CW' or UPmethod =='CW_C'):
                self.eta = 0.7  # in \eta in [0.5,1]
                self.a   = 1

            elif (UPmethod == 'AROW' or UPmethod =='AROW_C'):
                self.C = 1      # i.e., parameter r
                self.a = 1      # default
            
            
            elif (UPmethod == 'SOP' or UPmethod =='SOP_C'):
                self.a = 1

            elif (UPmethod == 'IELLIP' or UPmethod =='IELLIP_C'):
                self.b = 0.3
                self.IELLIP_c = 0.1   # c=0.5
                self.a = 1

            elif (UPmethod == 'SCW' or UPmethod =='SCW_C'):
                self.eta = 0.75
                self.C   = 1
                self.a   = 1

            elif (UPmethod == 'NAROW' or UPmethod =='NAROW_C'):
                self.C = 1         #i.e., parameter r
                self.NAROW_b = 1
                self.a = 1

            elif (UPmethod == 'NHERD' or UPmethod =='NHERD_C'):
                self.C = 1   # NHERD_C = C
                self.a = 1

            elif (UPmethod == 'NEW_ALGORITHM' or UPmethod =='NEW_ALGORITHM_C'):
                # initialie your parameters here...
                pass
            
            else:
                print('Unknown method.')

        elif (task_type == 'mc'):
            
            if (UPmethod == 'M_PERCEPTRON' or UPmethod =='M_ROMMA' or UPmethod == 'M_AROMMA' or 
            UPmethod =='M_PERCEPTRON_C' or UPmethod =='M_ROMMA_C' or  UPmethod =='M_AROMMA_C'):
                pass
            
            elif (UPmethod == 'M_PERCEPTRONU' or UPmethod =='M_PERCEPTRONU_C'):
                pass
            
            elif (UPmethod == 'M_PERCEPTRONS' or UPmethod =='M_PERCEPTRONS_C'):
                pass
            
            elif (UPmethod == 'M_OGD' or UPmethod =='M_OGD_C'):
                self.t = 1      # iteration no, learning rate eta_t = 1/sqrt(t)
                self.C = 1

            elif (UPmethod == 'M_PA' or UPmethod =='M_PA_C' or UPmethod == 'M_PA1' or 
            UPmethod =='M_PA2' or UPmethod =='M_PA1_C' or  UPmethod =='M_PA2_C'):
                self.C = 1


            elif (UPmethod == 'M_CW' or UPmethod =='M_CW_C'):
                self.eta = 0.75  # in \eta in [0.5,1]
                self.a   = 1

            elif (UPmethod == 'M_SCW1' or UPmethod =='MSCW_2' or UPmethod == 'M_SCW1_C' or UPmethod =='MSCW_2_C'):
                self.eta = 0.75
                self.C   = 1
                self.a   = 1

            elif (UPmethod == 'M_AROW' or UPmethod =='M_AROW_C'):
                self.C = 1      # i.e., parameter r
                self.a = 1      # default

            elif (UPmethod == 'NEW_ALGORITHM' or UPmethod =='NEW_ALGORITHM_C'):
                # initialie your parameters here...
                pass
