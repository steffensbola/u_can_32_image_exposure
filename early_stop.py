'''
Class to handle early stopping when working with batch-wise training
'''
class early_stop():
    def __init__ (self, thr = 0.001, tol = 200, in_final = False):
        self.thr = thr
        self.tol = tol
        self.in_final = in_final
        self.last_imp = 0
        self.last_loss = 999999999999.
        self.wait = 0
        self.save_flag = False
    
    '''
    Verifies if the training should stop and updates the current training values
    input: 
        epoch: integer
            current epoch counter
        loss_value: float
            current loss value
    output:
        stop: boolean
            boolean value that indicates if the training should be stopped.
    
    
    '''
    def stop(self, epoch, loss_value):
        self.save_flag = False
        if (self.last_loss - loss_value > self.thr):
            print("epoch: ", epoch,"loss_value: ", loss_value, "last loss: ", self.last_loss,"dif_los: ", self.last_loss - loss_value)
            self.last_imp = epoch
            self.last_loss = loss_value
            self.save_flag = True
            #print("epoch: ", epoch,"loss_value: ", loss_value, "last loss: ", self.last_loss,"dif_los: ", self.last_loss - loss_value)
            
        if(not self.in_final):
            if ((epoch - self.last_imp) >= self.tol):
                print('Time to stop the training!!')
                return True
        
        return False
    
    '''
    Verifies if the training has improved and if the model should be saved
    '''
    def save(self):
        return self.save_flag