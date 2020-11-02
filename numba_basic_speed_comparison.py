import numpy as np
import numba
from myownfcnet.numba_utils import numba_matmul, mean_list_numba
import time


def mean_list(x):
    sum_val = 0
    N = len(x)
    for i in range(N):
        sum_val += x[i]
    return sum_val / N

def mean_list_numpy(x):
#    x = np.asarray(x)
    return x.mean()    

def test_matmul_times():
    
    print("\n--------------------- MATMUL -------------------")

    # remove numba initial overhead for comparison
    
    x = np.array([[2,1],[2,3],[7,8]])
    y = np.array([[1,2],[3,4]])
    start = time.clock()
    res = numba_matmul(x, y) 
    end = time.clock()
    print('\nnumba-overhead-time: {}'.format(end-start))

    for x1,x2,y2 in [[   5,   7,    1],
                     [  50,  70,    1],
                     [  50,  70,   10], 
                     [ 500, 700,  100], 
                     [5000, 700, 1000],]:
        print(f"\nx1={x1}, x2=y1={x2}, y2={y2}")

        x = np.arange(x1*x2).reshape((x1,x2))
        y = np.arange(x2*y2).reshape((x2,y2))

        start = time.clock()
        result_np = np.matmul(x, y)
        end = time.clock()
        print('numpy:        {}'.format(end-start))

        # result_nb1 = np.zeros((x.shape[0], y.shape[1]))
        start = time.clock()
        result_nb = numba_matmul(x, y) #, result_nb1)
        end = time.clock()
        print('numba:        {}'.format(end-start))

        # # result_nb2 = np.zeros((x.shape[0], y.shape[1]))
        # start = time.clock()
        # result_nb2 = numba_matmul(x, y) #, result_nb2)
        # end = time.clock()
        # print('numba-second: {}'.format(end-start))

        assert np.all(result_np == result_nb) # and np.all(result_np == result_nb2)

def test_mean_times():

    print("\n---------------------- MEAN --------------------")

    num_inputs = 10 ** np.arange(2,7)
    print(num_inputs)

    # remove numba initial overhead for comparison
    start = time.clock()
    sum_val_numba = mean_list_numba(np.array([1,2])) 
    end = time.clock()
    print('\nnumba-overhead-time: {}'.format(end-start))


    for num_input in num_inputs:

        a = [0,1,2,3,4,5,6,7,8,9] * int(num_input)
        print('\nnumber of elements = {:.0e}'.format(len(a)))
        #    print(a)

        a = np.asarray(a)

        start = time.clock()
        sum_val_list = mean_list(a)
        end = time.clock()
        print(f'pure python: {end-start}')

        start = time.clock()
        sum_val_np = np.mean(a)
        end = time.clock()
        print(f'numpy:       {end-start}')

        start = time.clock()
        sum_val_numba = mean_list_numba(a)
        end = time.clock()
        print(f'numba:       {end-start}')
        #    mean_list_numba.inspect_types()

        assert sum_val_list == sum_val_np == sum_val_numba


def main():
    TEST_MATMUL_TIMES = True
    TEST_MEAN_TIMES = True


    if TEST_MATMUL_TIMES:
        test_matmul_times()
    if TEST_MEAN_TIMES:
        test_mean_times()


    
if __name__ == "__main__":
    main()

