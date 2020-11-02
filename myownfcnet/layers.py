import numpy as np

class FcLayer():
    def __init__(self, num_neurons, num_neurons_pre, activation_function):
        self.weights = np.random.random((num_neurons_pre, num_neurons))/10. - 0.05 
        self.biasses = np.random.random((num_neurons))/10. - 0.05

        if activation_function == 'relu':
            self.act_func = self.relu
            self.act_func_deriv = self.relu_deriv
        elif activation_function == 'softmax':
            self.act_func = self.softmax
            self.act_func_deriv = self.softmax_deriv
        else:
            raise ValueError('No activation function named {}'.format(activation_function))

    def forward(self, x):
        z = np.matmul(x, self.weights) + self.biasses 
        return self.act_func(z)

    def forward_train(self, x):
        z = np.matmul(x, self.weights) + self.biasses 
        return z, self.act_func(z)

    def relu(self, x):
        x[x<0] = 0.
        return x

    def relu_deriv(self, x, _):
        batchsize, N = x.shape 
        x_deriv = np.ones(x.shape)
        x_deriv[x<0] = 0.
        
        returnarr = np.empty((batchsize, N, N))
        for i in range(batchsize):
            returnarr[i] = np.multiply(np.identity(x.shape[1]), x_deriv[i])
        
        return returnarr

    def softmax(self, x):
        y = np.exp(x - np.max(x)) # - np.max(x) for numerically stable execution
        return y / np.sum(y, axis=1, keepdims=True) 

    def softmax_deriv(self, _, x): # TODO
        batchsize, N = x.shape 
        returnarr = np.empty((batchsize, N, N))
        for i in range(batchsize):
            returnarr[i] = np.multiply(np.identity(N), x[i]) - np.outer(x[i], x[i])
        return returnarr