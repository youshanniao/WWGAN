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
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from model import Generator, Discriminator
from Utils import image_processing as im


# Hyperparameters
ORDER = 1 # how many colums of the sample, 'the input_size'
# SAMPLE_SIZE = 1500 # how many raws in samples wirh the same dirtribution
SLICE = 25 # the smallist slice of sample
DIM = 67# the number of hidden nodes of nets, the 'hidden_size'
LR = 1e-4 # learning rate of the Adam optimizator, bigger the faster the operater restrain, but accuraty decay
EPOCH = 600 # how many [G and D] iterations to train for, the basic number
BATCH_SIZE = 13 # batch size of eaach dataloader, how many samples for one CRITIC_ITER
CRITIC_ITERS = 5 # hom many critic iterations pre D iteration
LAMBDA = .01 # 0.1-10, can be changed to suit the model
THRESHOLD = 0.2 # the threshold to devide the whether the distributions are the same
BETA1 = 0.1 # first beta of Adam optimization
BETA2 = 0.999 # second beta of Adam optimization
CUT = 5 # the number of critic inputs

# define the theme of seaborn to drow images
sns.set_theme(palette="deep", style='ticks', color_codes=True, font='Times New Roman', font_scale=1.5)

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
slicedata = np.empty([0,2])
cyclenum = 0
for i in NUM:
    iternum = np.loadtxt('data' + '/' + str(seed) + '_' + 'slice' + '_' + str(int(i)) + '.csv')
    iternum = iternum.reshape(np.size(iternum), 1)
    print('mean=' + str(np.mean(iternum)))
    print('std=' + str(np.std(iternum)))
    cyclenum += np.size(iternum)
    slicenum = np.full((np.size(iternum),1), cyclenum)
    slicedata_i = np.concatenate((iternum, slicenum), axis=1)
    slicedata = np.append(slicedata, slicedata_i, axis=0)
sliced = pd.DataFrame(data=slicedata, columns=['Value', 'Time'])
print(sliced)    

# drow the SwamPlot with seaborn
f, cy = plt.subplots(1, 1, figsize=(30, 6))
sns.violinplot(data=sliced, x='Time', y='Value', inner=None)
sns.swarmplot(data=sliced, x='Time', y='Value', color='white', edgecolor='gray')
cy.grid(axis='y')
plt.savefig('SwamPlot', dpi=300)
plt.close()


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

noise_show = torch.randn(np.size(real_data_slice), ORDER)
noise_show=noise_show.detach().numpy()

# create the Panda dataform to drow the figs
n = np.concatenate((cycles, real_dist, fake_dist_add), axis=1) # remeber this!!!!
data_fix = pd.DataFrame(data=n, columns=['Time', 'Real Samples', 'Fake Samples'])

# plot the real/fake data
f, g = plt.subplots(1, 1, figsize=(12, 6))
g = sns.scatterplot(data=data_fix, x='Time', y='Real Samples', color='r')
plt.savefig('Real', dpi=300)
plt.close()

f, g = plt.subplots(1, 1, figsize=(12, 6))
g = sns.scatterplot(data=data_fix, x='Time', y='Fake Samples', color='b')
plt.savefig('Fake', dpi=300)
plt.close()

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
