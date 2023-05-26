# -*- coding: utf-8 -*-
# The verification of the WWGAN au gmentation rsults for one-dimensional time-series data
# !!!!! RUN THE WWGAN_toy.py FIRST !!!!!
#
# ================================================= What does this code do ====================================================
# Verify the data augmentation results by drowing contrasting figures
# Output: Real.png, Fake.png, SwamPlot.png, Real & Fake.png, Fitting.png
# Learning the time-varying distribution of the time-series sample, construct and save the Generator, output: /models/modelG_X_X
# Devide the  input time-series into slices that contain the same distribution, output: /data/X_slice_X.csv
# Save each slice's loss figure and distribution figure, output: /images/_X_X.png
# Drow a figure of the generated and original data of each silce, output: /Result_X.png

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from model import Generator, Discriminator
from Utils import image_processing as im
from Utils import data_processing as da


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

# Special parameter 
NUM = np.loadtxt('data/index_J.csv') # the index of slices
print(NUM)

# the parameters needed
seed = 1
indexG = {}
indexD = {}
fake_dist_add = []

# load the original data and reshape it
real_data = np.loadtxt('data/sample.csv')
real_dist = real_data.reshape(np.size(real_data), 1)
sliced = da.slice_concat(seed, NUM)

image1 = im.image(ORDER, DIM, BATCH_SIZE, EPOCH, seed)

# creat the FAKE data with saved netG
for index in NUM: 

    # initial the shapes of netG and netD
    netG_l = Generator(ORDER, DIM, ORDER)
    netD_l = Discriminator(ORDER, DIM, ORDER)

    # pump out the index of tthe state dict of netG and ndtD
    indexG[index] = 'models' + '/' + 'modelG' + '_' + str(seed) + '_' + str(int(index))
    indexD[index] = 'models' + '/' + 'modelD' + '_' + str(seed) + '_' + str(int(index))

    # load the weights of netG and netD
    netG_l.load_state_dict(torch.load(indexG[index]))
    netG_l.eval()
    netD_l.load_state_dict(torch.load(indexD[index]))
    netD_l.eval()

    # load the sliced data
    real_data_slice = np.loadtxt('data' + '/' + str(seed) + '_'  +'slice' + '_' + str(int(index)) + '.csv')
    print(np.size(real_data_slice))

    noise = torch.randn(np.size(real_data_slice), ORDER)
    fake_data = netG_l(noise)
    fake_dist = fake_data.data.numpy()
    fake_dist_add.extend(fake_dist)
    
print(np.size(fake_dist_add))

# creat the number of charge-discharge cycle data
cycles = np.empty([np.size(real_dist),1])
cycles[:,0] = np.array(range(1, np.size(real_dist)+1, 1))
print(np.size(cycles))

# create the Panda dataform to drow the figs
n = np.concatenate((cycles, real_dist, fake_dist_add), axis=1) # remeber this!!!!
data_fix = pd.DataFrame(data=n, columns=['Time', 'Real Samples', 'Fake Samples'])

# change the wide-form data to long-form data
data_trans = data_fix.melt(
    id_vars=['Time'],
    var_name='Types',
    value_name='Value'
)
print(data_trans)

# plot the Real & Fake fig
f, g = plt.subplots(1, 1, figsize=(12, 6))
g = sns.lineplot(data=data_trans, x='Time', y='Value',hue='Types',)
plt.savefig('Real & Fake', dpi=300)
plt.close()

# plot the real and fake fitting
sns.set_theme(palette="deep", style='whitegrid', color_codes=True, font='Times New Roman', font_scale=1.5)
rf = sns.lmplot(data=data_trans, x='Time', y='Value',hue='Types', ci=100, truncate=False, robust=True, palette=['b', 'r'], markers=['o', 'x'], height=6, aspect=2)
rf.tight_layout()
rf.savefig('FinalOut', dpi=300)
