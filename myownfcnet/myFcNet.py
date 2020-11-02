#TODO s:
# - implement more optimizers and the corresponding class structure
# - implement evaluation while training
# - implement more metrics for evaluation and the corresponding class structure 
# - implement different loss functions


import numpy as np
import numba
import time

from myownfcnet.layers import FcLayer
from myownfcnet.numba_utils import numba_matmul
            
##-------------------------------- optimizers -------------------------------------
#class optimizers(): #TODO implement multiple optimizers
#    def __init__(self, optimizer, layer):
#        
#        if optimizer == 'SDG':
#            self.optimizer = self.SDG
#        else:
#            raise ValueError('No activation function named {}'.format(activation_function))

#    
#    def gradient_descent(self, learning_rate):
        

class FcNet():
    '''
    fully connected neural network with relu or softmax activation functions and stochastic gradient decent as optimizer
    ''' #TODO loss functions

    def __init__(self, list_num_neurons_per_layer, list_activations, use_numba: bool = True): 
        self.train_bool = False

        self.num_layers = len(list_num_neurons_per_layer)
        self.num_neurons = list_num_neurons_per_layer
        self.activations = list_activations
        self.use_numba = use_numba

        self.layers = []
        for i in range(1, self.num_layers):
            self.layers.append(FcLayer(self.num_neurons[i], 
                                       self.num_neurons[i-1],
                                       self.activations[i-1]))

    @property
    def use_numba(self):
        return self._use_numba

    @use_numba.setter
    def use_numba(self, use_numba: bool):
        self.batch_loop = self.numba_batch_loop if use_numba else self.numpy_batch_loop
        self._use_numba =use_numba

    def forward(self, x): 
        if self.train_bool: 
            # saves the nodevalues of the forward loop for training (loss_calc and
            # back_pass)
            self.current_nodevalues[0] = x
            for i,layer in enumerate(self.layers):
                self.current_unactivated_values[i+1], self.current_nodevalues[i+1] = layer.forward_train(self.current_nodevalues[i])
        else:
            for layer in self.layers:
                x = layer.forward(x)
            return x            


    def reshape_yArr(self, y_single_labels):
        y_train_arr = np.zeros((y_single_labels.shape[0], self.num_neurons[-1]))
        for i,row in enumerate(y_train_arr):
            row[int(y_single_labels[i])] = 1
        return y_train_arr

    def train(self, x_train, y_train, num_epochs, batchsize, learning_rate, optimizer, loss_func, shuffle=True):
        print('')
        print('Begin Training')
        self.train_bool = True
        self.current_nodevalues = [None] * self.num_layers
        self.current_unactivated_values = [None] * self.num_layers
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        num_traindata = x_train.shape[0]
        num_batches = num_traindata // batchsize
        shuffle_mask = np.arange(num_traindata)

        #check for y_train shape in case of classification labels
        if x_train.shape[0]==y_train.shape[0]:
            if (len(y_train.shape) == 1) and (x_train.shape[1] != 1):
                
                y_train_arr = self.reshape_yArr(y_train)
            else:
                y_train_arr = y_train
        else:
            raise ValueError("x_train and y_train don't have same amout of training data")

        # training loop        
        for epoch in range(num_epochs):
            if shuffle:
                np.random.shuffle(shuffle_mask)

            x_epoch = x_train[shuffle_mask,...]
            y_epoch = y_train_arr[shuffle_mask,...]
            losss = []
            for i in range(num_batches):
                x_batch = x_epoch[i*batchsize:(i+1)*batchsize]
                y_batch = y_epoch[i*batchsize:(i+1)*batchsize]
                self.forward(x_batch)
                losss.append(self.calc_loss(y_batch))
                self.back_pass(y_batch)
            print('loss = {}'.format(np.asarray(losss).mean()))
            print('epoch {} complete'.format(epoch))
        
        self.train_bool = False


    def calc_loss(self, y_batch):
        y_pred = self.current_nodevalues[-1]
        if self.loss_func == 'MSE':
            loss = 0.5 * np.sum((y_pred - y_batch)**2, axis=1) 
        elif self.loss_func == 'cross_entropy':
            loss = None #TODO
        else:
            raise ValueError('No loss function named {}'.format(self.loss_func))

        return np.mean(loss, axis=0)

    @staticmethod
    @numba.jit(nopython=True)
    def numba_batch_loop(batchsize, derivLossZb, derivLossYb, derivYbZb, derivLossWab, derivZbWab):
        for i in range(batchsize):
            derivLossZb[i] =  numba_matmul(derivLossYb[i], derivYbZb[i])       
            derivLossWab[i] = np.outer(derivZbWab[i,:], derivLossZb[i,:])

    @staticmethod
    def numpy_batch_loop(batchsize, derivLossZb, derivLossYb, derivYbZb, derivLossWab, derivZbWab):
        for i in range(batchsize):
            derivLossZb[i] =  np.matmul(derivLossYb[i], derivYbZb[i])       
            derivLossWab[i] = np.outer(derivZbWab[i,:], derivLossZb[i,:])
    
    def back_pass(self, y_batch):

        batchsize = y_batch.shape[0]
        #TODO include other loss functions than mse. only has effect in this line
        derivLossYb = np.expand_dims((self.current_nodevalues[-1] - y_batch), axis=1) 
        for i,layer in enumerate(reversed(self.layers)):
            
            derivZbWab = self.current_nodevalues[-(i+2)]
            derivYbZb = layer.act_func_deriv(self.current_unactivated_values[-(i+1)], self.current_nodevalues[-(i+1)])
    
            derivLossZb = np.empty((batchsize, derivYbZb.shape[2]))
            derivLossWab = np.empty((batchsize, derivZbWab.shape[1], derivLossZb.shape[1]))

            self.batch_loop(batchsize, derivLossZb, derivLossYb, derivYbZb, derivLossWab, derivZbWab)

            derivLossWab = np.mean(derivLossWab, axis=0)
            derivLossBa = np.mean(derivLossZb, axis=0 ) # * identity as partial z/partial b = identity

#            optimizer = optimizers()
#            optimizer.optimizer(learning_rate, layer, delLossWab, delLossBab) #TODO implement way to change optimizers

            layer.weights -= self.learning_rate * derivLossWab
            layer.biasses -= self.learning_rate * derivLossBa 

            derivLossYb = np.matmul(derivLossYb, layer.weights.T) #safe derivate for next layer calculation
           

    def predict_on_batch(self, x):
        return self.forward(x)
        

    def predict(self, x):
        x_batch = x[None, ...]
        y_batch = self.predict_on_batch(x_batch)
        print(y_batch.shape)
        print(y_batch)
        return y_batch[0,:]

    def evaluate(self, x_test, y_test): #TODO specified for metric='accuracy'
        #y_test is 1-D array with correct index_label (not full probability distribution)
        # TODO let y_test be full prob distribution or at least 2D arr
        print('')
        print('evaluate')
        y_pred = np.argmax(self.forward(x_test), axis=1)
        temp = np.zeros(y_test.shape[0])
        temp[y_pred==y_test] = 1.
        accuracy = temp.mean()
        return accuracy


if __name__ == '__main__':    
    np.random.seed(1)

    batchsize = 32

    num_data = 1000
    test_data_len = 28*28
    output_len = 10

    model = FcNet([test_data_len, 100, 20, output_len], ['relu', 'relu', 'softmax'])
   
#    x = np.ones((batchsize,test_data_len))
#    outp = model.predict_on_batch(x)
#    print(type(outp))
#    print(outp)

#    x_single = np.ones(4)
#    outp_single = model.predict(x_single)
#    print(type(outp_single))
#    print(outp_single)




    y_train = np.zeros((num_data, output_len))
    mask = np.stack((np.arange(num_data), np.random.randint(output_len, size=(num_data))), axis=-1) 
    y_train[mask] = 1 

#    y_train_single = np.random.randint(10, size=(1000))

    model.train(x_train=np.random.random((num_data,test_data_len)), 
                y_train= np.ones((num_data)), 
                num_epochs=10, batchsize=batchsize, optimizer='SDG',
                learning_rate=0.001, loss_func='MSE')

#    list_ = [1,2,3,4]
#    list_rev = reversed(list_)
#    print(type(list_rev))
#    print(list_rev)
#    for i in list_rev:
#        print(i)









    
