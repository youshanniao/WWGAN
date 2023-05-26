# -*- coding: utf-8 -*-
# The WWGAN for one-dimensional time-series data augmentation
#
# ===================================================== The toy dataset ======================================================
#
# The real dataset is built upon a Gamma process: Gamma(α, β). In which α = 12 * t^0.4 is a variable, β = 0.2 is a constant
# The α in Gamma process shifts 5 times, and each time produce n sapmles, n is a variable which n~Gaussian(25, 4)
#
# ================================================= What does this code do ====================================================
# Learning the time-varying distribution of the time-series sample, construct and save the Generator, output: /models/modelG_X_X
# Devide the  input time-series into slices that contain the same distribution, output: /data/X_slice_X.csv
# Save each slice's loss figure and distribution figure, output: /images/_X_X.png
# Drow a figure of the generated and original data of each silce, output: /Result_X.png


import numpy as np
import torch
import time
import model
from Utils import image_processing as im

# Hyperparameters
ORDER = 1 # how many colums of the sample, 'the input_size'
SLICE = 25 # the slice window of sample
DIM = 67 # the number of hidden nodes of nets, the 'hidden_size'
LR = 1e-4 # learning rate of the Adam optimizator, bigger the faster the operater restrain, but accuraty decay
EPOCH = 600 # how many [G and D] iterations to train for
BATCH_SIZE = 13 # batch size of eaach dataloader, how many samples for one CRITIC_ITER
CRITIC_ITERS = 5 # hom many critic iterations pre D iteration
LAMBDA = .01 # resommend 0.01-10, can be changed to suit the model
THRESHOLD = 0.2 # the threshold to devide the whether the distributions are the same
BETA1 = 0.1 # first beta of Adam optimization
BETA2 = 0.999 # second beta of Adam optimization
CUT = 5 # the number of critic inputs

# =====================
# (1) creat the sample
# =====================

# creat the sample, 50000 Gaussian distribution with initial mu=3+(3^0.25/100)*t, sigma=0.5
alpha = 12.
beta = .2

bath = [[0]]
for t in range(1,6):
    bath_in = np.random.gamma(alpha * t ** .4, beta, (np.abs(int(np.random.normal(25,4))), ORDER))
    bath = np.append(bath, bath_in, axis=0)
    
sample = bath

# to cut the sample to silces
def cut_slice(data, begin, end):
    n = data[begin : end, :] # cut a slice of 100 data
    data = n.astype('float32') # numpy
    dataset = torch.from_numpy(data) # tensor
    return dataset, data

# ================================Let the Canon Begin!========================================

# to record the bigining time of loop
start = time.time()

# =====================
# (2) difine variables
# =====================

# the dics to record parameters within the loop
loop = {}
w_distance_s = {}
slice_data = {}
index_G = {}
index_D = {}

# the seed is kind of a HYPER iter, which used to try different loop parameters
seed = 1

# the iters in the loop
i = SLICE - 1
j = 0
index_J = []


# ======================
# (3) training & testing
# ======================

# how many loops to try
while seed <= 1:

    loop[seed] = model.Loop(ORDER, DIM, LR, BATCH_SIZE, LAMBDA, EPOCH, THRESHOLD, 
                            index_G, index_D, CRITIC_ITERS, CUT, BETA1, BETA2, seed)
    while j <= np.size(sample)//ORDER:
        # train the first silce
        if w_distance_s == {}:
            train_data, _ = cut_slice(sample, i - SLICE + 1, i )
            index_G, index_D, w_distance_s[j], w_costs = loop[seed].train_loop(train_data, j)
            slice_data[j] = _

        # the W distance of the current slice does not meet the quality
        elif w_distance_s[j] >= THRESHOLD:
            train_data, _ = cut_slice(sample, i - SLICE + 1, i  )
            index_G, index_D, w_distance_s[j], w_costs = loop[seed].train_loop(train_data, j)
        
        # use the trained netG and netD to test the data from next slice
        else:
            k = 0
            if i == np.size(sample)//ORDER:
                testing_data, dataset_slice = cut_slice(sample, j, i)
                w_distance_next = loop[seed].test(index_G, index_D, j, testing_data)

                if w_distance_next >= THRESHOLD:
                    _, slice_data[j] = cut_slice(sample, j, i - SLICE + 1)
                    index_J = np.append(index_J, j)
                    j = i - SLICE + 1
                    w_distance_s[j] = w_distance_next

                else:
                    _, slice_data[j] = cut_slice(sample, j, i)
                    index_J = np.append(index_J, j)
                    break

            else:
                i = i + 1
                testing_data, dataset_slice = cut_slice(sample, i - SLICE + 1, i )
                w_distance_next = loop[seed].test(index_G, index_D, j, testing_data)
            

                if w_distance_next >= THRESHOLD:
                    _, slice_data[j] = cut_slice(sample, j, i - SLICE + 1)
                    index_J = np.append(index_J, j)
                    j = i - SLICE + 1
                    w_distance_s[j] = w_distance_next

                else:
                    w_distance_s[j] = w_distance_next
                
        
    # before ending the loop, show the index of iters
    print('i=' + str(i))
    print('j=' + str(j))
    print('k=' + str(k))

    # save the original data and the silced data
    print('sample data has been saved to /data/sample.csv')
    np.savetxt('data/sample.csv', sample, delimiter=',')
    print('sliced data has been saved to /data/slice_data[x].csv')
    print(index_J)
    np.savetxt('data/index_J.csv', index_J, fmt='%d', delimiter=',')
    # print(slice_data)
    

    for p in index_J:
        np.savetxt('data' + '/' + str(seed) + '_'  +'slice' + '_' + str(int(p)) + '.csv', slice_data[p], delimiter = ',')
    
    # clear the w_distance_s for next loop
    w_distance_s = {}
    seed = seed + 1
# record the ending time of the loop and print the time cost
end = time.time()
print('Runing time: %.8s s' % (end-start))


NUM = np.loadtxt('data/index_J.csv')
image1 = im.image(ORDER, DIM, BATCH_SIZE, EPOCH, seed=1)
image1.real_fake_image(NUM)
