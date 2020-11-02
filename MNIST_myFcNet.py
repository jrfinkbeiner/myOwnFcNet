import time
import numpy as np
from tensorflow.keras.datasets import mnist # as tf # to load MNIST-dataset
import matplotlib.pyplot as plt

import myownfcnet.myFcNet as myFcNet

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def create_plots(N, x_test, y_test, predictions, class_names):
    num_labels = predictions.shape[1]
    plt.figure()
    for i in range(N):
        plt.subplot(N,2,2*i+1)
        plt.imshow(x_test[i].reshape(28,28))

        plt.subplot(N,2,2*i+2)
        plot_value_array(i, predictions, y_test)
        plt.xticks(range(num_labels), class_names, rotation=45)
    plt.show()



def main():
    print('\n-----------------------------------------------------------------------------')
    print('Training a simple two-layer feed-forward neural network on the MNIST-dataset.\n')
    print('Training for one epoch takes approximately 1-2min.\n')

    np.random.seed(1)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    num_labels = np.max(y_train) + 1

    x_train_flat =  x_train.reshape((x_train.shape[0], -1))
    x_test_flat = x_test.reshape((x_test.shape[0], -1))

    num_inputs = x_train_flat.shape[1]

    # #---- numpy ----    
    # model = myFcNet.FcNet([num_inputs, 100, num_labels], ['relu', 'softmax'], use_numba=False)

    # batchsize=32
    # start = time.clock()
    # model.train(x_train=x_train_flat, 
    #             y_train=y_train, 
    #             num_epochs=1, 
    #             batchsize=batchsize, 
    #             optimizer='SDG',
    #             learning_rate=0.0001, 
    #             loss_func='MSE',)
    # end = time.clock()
    # print('\ntime (numpy):    {}'.format(end-start))


    #---- numba ----    
    model = myFcNet.FcNet([num_inputs, 100, num_labels], ['relu', 'softmax'], use_numba=True)

    batchsize=32
    start = time.clock()
    model.train(x_train=x_train_flat, 
                y_train=y_train, 
                num_epochs=1, 
                batchsize=batchsize, 
                optimizer='SDG',
                learning_rate=0.0001, 
                loss_func='MSE',)
    end = time.clock()
    print('time (numba):    {}\n'.format(end-start))

    accuracy = model.evaluate(x_test_flat, y_test)

    print('accuracy = {:.2f}%'.format(accuracy*100))

    y_pred = model.predict_on_batch(x_test_flat)

    num_plots = 10
    create_plots(num_plots, x_test, y_test, y_pred, class_names)






if __name__ == "__main__":
    main()
