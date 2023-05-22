# 2022/06/26 bulit for generate time varing samples of wind power ELM_Mslp_Try

# =================================================The Theroy==================================================
# The real data contains 2000 samples of tested pressure from 4 sensors
# 

# Using WGAN-GP to generate fake samples and critic whether the distribution of real samples has been switche


import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import autograd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd
from scipy import io

# Hyperparameters
ORDER = 1 # how many colums of the sample, 'the input_size'
# SAMPLE_SIZE = 1500 # how many raws in samples wirh the same dirtribution
SLICE = 12 # the smallist slice of sample
DIM = 42 # the number of hidden nodes of nets, the 'hidden_size'
LR = 1e-4 # learning rate of the Adam optimizator, bigger the faster the operater restrain, but accuraty decay
EPOCH = 1000 # how many [G and D] iterations to train for, the basic number
BATCH_SIZE = 6 # batch size of eaach dataloader, how many samples for one CRITIC_ITER
CRITIC_ITERS = 5 # hom many critic iterations pre D iteration
LAMBDA = .01 # 0.1-10, can be changed to suit the model
THRESHOLD = 0.2 # the threshold to devide the whether the distributions are the same
BETA1 = 0.1 # first beta of Adam optimization
BETA2 = 0.999 # second beta of Adam optimization
CUT = 5 # the number of critic inputs

# load the real samples
data = pd.read_csv('data/data37.csv')

# sellect 2000 of the samples as training samples
data_se = data.loc[:, ['CCCT']]
dataset = data_se.to_numpy()
print(np.size(dataset))

# define the theme of seaborn to drow images
sns.set_theme(palette="deep", style='ticks', color_codes=True, font='Times New Roman', font_scale=1)

# ======================================Defination Process=====================================
# construct the generator with 2 hidden linear layers followed by ReLU
class Generator(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Hardshrink(),
            nn.Linear(hidden_size, hidden_size),
            nn.Hardshrink(),
            nn.Linear(hidden_size, hidden_size),
            nn.Hardshrink(),
            nn.Linear(hidden_size, hidden_size),
            nn.Hardshrink(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, noise):
        output = self.linear_relu_stack(noise)
        return output

# construct the discriminator with 2 hidden linear layers followed by ReLU
class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, inputs):
        output = self.linear_relu_stack(inputs)
        return output

# construct the Loop class so that one can write loops with different value of hyperparameters
class Loop:
    def __init__(self, order_in, dim_in, lr_in, batch_size_in, lambda_in, indexG_in, indexD_in, beta1_in, beta2_in, seed_in):

        self.order = order_in
        self.dim = dim_in
        self.Lr = lr_in
        self.batch_size = batch_size_in
        self.lamba = lambda_in
    
        # the list that save the pass of nets' weights, NOT net itself
        self.indexG = indexG_in
        self.indexD = indexD_in

        # the BETAs of Adam optimization
        self.beta1 = beta1_in
        self.beta2 = beta2_in

        # the index of different Loop
        self.seed = seed_in

    # the training process of netG and netD
    def train_loop(self, epoch_in, critic_iters_in, training_data_in, index_in):
        
        epoch = epoch_in
        critic_iters = critic_iters_in

        # the sliced data
        training_data = DataLoader(training_data_in, batch_size=self.batch_size, shuffle=True)

        # index for saving the images and model weights
        index = index_in

        # bulid the netG and netD in the loop
        self.netG = Generator(self.order, self.dim, self.order)
        self.netD = Discriminator(self.order, self.dim, self.order)

        # weight initialization
        self.netG.apply(self.weights_init)
        self.netD.apply(self.weights_init)
        print(self.netG)
        print(self.netD)

        # set the optimizer for netG and netD
        optimizerG = torch.optim.Adam(self.netG.parameters(), lr=self.Lr, betas=(self.beta1, self.beta1))
        optimizerD = torch.optim.Adam(self.netD.parameters(), lr=self.Lr, betas=(self.beta1, self.beta2))

        # creat to 1 tensor for backward(), 
        # tensor[1] means the gradients are positive and the optimize will minimum the gradients
        # tensor[-1] means the gradients are negative and the optimize well maxmum the gradients
        one= torch.tensor(1, dtype=torch.float)
        m_one = one * -1

        # three dists to record the Wasserstein distance(w_cost), 
        # the critic loss of real data(r_cost), and the loss of fake data(g_cost)
        w_cost = []
        r_cost = []
        g_cost = []

        # the traing loop
        for iteration in range(epoch):
            
            # =================
            # (1) Update netD
            # =================

            # free the gradient of netD
            for p in self.netD.parameters(): 
                p.requires_grad = True

            # the iteration of netD tring
            for iter_d in range(critic_iters):
                
                # load the data from Dataloader, the batch size is defiend by BATCH_SIZE 
                _data = next(iter(training_data))
                real_data = _data
                
                # zero the gradient otherwise it will keep adding 
                self.netD.zero_grad()
                
                # critic the real data
                real_D = self.netD(real_data)
                real_D = real_D.mean()

                # critic the fake data
                noise = torch.randn(self.batch_size, self.order)
                _fake = self.netG(noise)
                fake_data = _fake
                fake_D = self.netD(fake_data)
                fake_D = fake_D.mean()

                # train with gradient penalty
                # noise = torch.randn(self.batch_size, self.order)
                # fake_data = self.netG(noise)
                gradient_penalty = self.calc_gradient_penalty(self.netD, real_data, fake_data)

                # calculate the loss and backward
                D_cost = fake_D - real_D + gradient_penalty
                D_cost.backward()
                # Wasserstein_D = real_D - fake_D
                optimizerD.step()

            # ===================
            # (2) Update netG
            # ===================

            # freeze the gradients in netD to avoid computation
            for p in self.netD.parameters():
                p.requires_grad = False

            self.netG.zero_grad()

            # the loss and backward
            noise = torch.randn(self.batch_size, self.order)
            fake_data = self.netG(noise)
            loss = self.netD(fake_data)
            loss = loss.mean()
            loss.backward(m_one)
            optimizerG.step()
            
            # =========================================
            # (3) Re-calculate the loss and save images
            # =========================================

            # the loss of netG
            G_cost = -loss
            g_distance = G_cost.detach().numpy()
            g_cost.append(g_distance)
            
            # the loss of netD
            real_D = self.netD(real_data)
            real_D = real_D.mean()
            r_distance = real_D.detach().numpy()
            r_cost.append(r_distance)

            # the loss of W distance
            Wasserstein_D = real_D + G_cost
            w_distance = abs(Wasserstein_D.detach().numpy())
            w_cost.append(w_distance)
            
            # tried to break from the epoch loop, but the results are bad
            # NEED more POWER!!!!
            '''
            if iteration > 200 and 0.5 * (w_cost[iteration] + w_cost[iteration - 1]) < THRESHOLD:
                self.creat_image(self.netG, _data, index)
                print('W distance: ' + str(w_distance))
                plt.plot(w_cost, label='W')
                plt.plot(g_cost, label='G')
                plt.plot(r_cost, label='D')
                plt.savefig('images' + '/' + '_' + str(self.seed) + '_' + 'Cost' + str(index))
                break
            '''
            
        # save the state dict(weights) of current netG and netD
        torch.save(self.netG.state_dict(), 'models' + '/' + 'modelG' + '_' + str(self.seed) + '_' +str(index))
        torch.save(self.netD.state_dict(), 'models' + '/' + 'modelD' + '_' + str(self.seed) + '_' +str(index))

        # pump out the index of tthe state dict of netG and ndtD
        self.indexG[index] = 'models' + '/' + 'modelG' + '_' + str(self.seed) + '_' + str(index)
        self.indexD[index] = 'models' + '/' + 'modelD' + '_' + str(self.seed) + '_' + str(index)
        indexG = self.indexG
        indexD = self.indexD
        
        # save an image
        self.creat_image(self.netG, _data, index, w_cost)

        # calculate the mean of last 5 w distence for output
        print(w_cost[-CUT:])
        w_dist = np.mean(w_cost[-CUT:])
        print('W distance: ' + str(w_dist))
        
        return indexG, indexD, w_dist, w_cost
        
    # the test loop to identify the switch of distribution
    def test(self, indexG_in, indexD_in, index_in, creitic_iters_in, old_data, test_data_in):

        index = index_in
        indexG = indexG_in
        indexD = indexD_in
        critic_iters = creitic_iters_in

        test_data = DataLoader(test_data_in, batch_size=self.batch_size, shuffle=True)

        # initial the shapes of netG and netD
        netG_l = Generator(self.order, self.dim, self.order)
        netD_l = Discriminator(self.order, self.dim, self.order)

        # load the weights of netG and netD
        netG_l.load_state_dict(torch.load(indexG[index]))
        netG_l.eval()
        netD_l.load_state_dict(torch.load(indexD[index]))
        netD_l.eval()

        # the optimizer for enhance train the netD to increase its critic ability
        optimizerD = torch.optim.Adam(self.netD.parameters(), lr=self.Lr, betas=(self.beta1, self.beta2))

        # creat to 1 tensor for backward()
        one= torch.tensor(1, dtype=torch.float)
        m_one = one * -1
        
        # entance train the netD with new data
        for p in self.netD.parameters(): 
                p.requires_grad = True
        
        new_dist = []

        for i in range(EPOCH):

            for iter_d in range(critic_iters):
                _data = next(iter(test_data))
                real_data = _data
                
                self.netD.zero_grad()
                
                real_D = self.netD(real_data)
                real_D = real_D.mean()

                # _data = next(iter(test_data))
                # real_data = _data

                # real_DD = self.netD(real_data)
                # real_DD = real_DD.mean()
                
                real_cost = real_D
                real_cost.backward(m_one)
                optimizerD.step()
            
            new_D = real_cost.detach().numpy()
            new_dist.append(new_D)
            cr = CRITIC_ITERS // 2
            mean_new = np.std(new_dist[-cr:])
            
            if i > CUT and mean_new < THRESHOLD:
                break
        
        for p in self.netD.parameters(): 
                p.requires_grad = False
        
        # calculate the W distance 5 times and return the mean
        ii = 1
        w_dist = []
        while ii <= CRITIC_ITERS:

            _data = next(iter(test_data))
            real_data = _data

            # Critic the real data
            real_D = netD_l(real_data)
            real_D = real_D.mean()

            # Critic the fake data
            noise = torch.randn(self.batch_size, self.order)
            _fake = netG_l(noise)
            fake_data = _fake
            fake_D = netD_l(fake_data)
            fake_D = fake_D.mean()

            # calculate the abs || of W distance
            Wasserstein_D = real_D - fake_D
            w_distance = abs(Wasserstein_D.detach().numpy())
            w_dist.append(w_distance)
            mean_w = np.mean(w_dist)
            ii = ii + 1
        
        # tried to use old data and new data to calculate the W distance, the result is bed
        # NEED MORE POWER!!!!
        '''
        _data = next(iter(old_data))
        real_data = _data
        real_D = netD_l(real_data)
        real_D = real_D.mean()

        _data = next(iter(test_data))
        new_data = _data
        new_D = netD_l(new_data)
        new_D = new_D.mean()

        Wasserstein_D = real_D - new_D
        w_distance = abs(Wasserstein_D.detach().numpy())
        print('test result: ' + str(w_distance))
        # w_dist.append(w_distance)
        # mean_w = np.mean(w_dist)
        '''
        
        # the five calculated W distance
        print(w_dist)
        # print the mean of W distance
        print('test result: ' + str(mean_w))
        return mean_w

    # weight initialization function
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, mode='fan_in')
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    # gradient penalty 
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(self.batch_size, 1)

        interpolates = alpha * real_data + ((1-alpha) * fake_data)

        disc_interpolates = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, 
                                    grad_outputs=torch.ones(
                                        disc_interpolates.size()),
                                    create_graph = True, retain_graph = True, only_inputs = True)[0]

        gradients_norm2 = gradients.norm(2, dim=1) # ord=2 means 2-norm, dim=1 means sum by column
        gradients_penalty = ((gradients_norm2 - 1) ** 2).mean() * self.lamba

        return gradients_penalty

    # Generate and save a plot of the true distribution, the generator, and the cirtic(didcriminator)
    def creat_image(self, netG_in, real_data, frame_index, w_cost_in):
      
        netG_image = netG_in
        index_image = frame_index
        w_cost = w_cost_in

        # use random normal noise to activate the netG
        noise = torch.randn(self.batch_size, self.order)
        # change tensor to numpy array 
        real_dist = real_data.numpy() 

        # generated fake samples
        fake_data = netG_image(noise)
        fake_dist = fake_data.data.numpy()

        # plot a 1 * 3 picture to show the generated data and the real data
        plt.clf()
        
        # plot an image with 3 axs
        fig, axs  = plt.subplots(1, 3, figsize=(15, 5))
        plt.subplots_adjust(wspace=0.2)
        # ax0 is the W loss
        axs[0].set_title('Wasserstein Distances', weight='bold')
        sns.lineplot(data=w_cost, ax=axs[0], estimator='mean', lw=2, color='steelblue')
        axs[0].set_xlabel('Iteration')
        axs[0].set_ylabel('Loss')
        
        # ax1 is the distributions of Real and Fake
        # axs[1].set_ylim([0,0.5])
        # axs[1].set_xlim([-15,25])
        axs[1].set_title('Real & Fake Distributions', weight='bold')
        sns.kdeplot(real_dist.flatten(), fill=True, ax=axs[1], label='Real data', color='b', bw_adjust=4)
        sns.kdeplot(fake_dist.flatten(), fill=True, ax=axs[1], label='Fake data', color='r', bw_adjust=4)
        axs[1].set_xlabel('Sample Value')
        axs[1].set_ylabel('Probability density')
        axs[1].legend(loc='upper right', frameon=True)
        
        # ax2 is the samples of Real and Fake
        # axs[2].set_ylim([0,8])
        # axs[2].set_xlim([0,8])
        axs[2].set_title('Real & Fake Samples', weight='bold')
        sns.histplot(x=real_dist.flatten(), y=fake_dist.flatten(), cmap='rocket_r', cbar=True)
        sns.kdeplot(x=real_dist.flatten(), y=fake_dist.flatten(), ax=axs[2])
        axs[2].set_xlabel('Real Samples')
        axs[2].set_ylabel('Fake Samples')

        plt.savefig('images' + '/' + '_' + str(self.seed) + '_' + str(index_image), dpi=300)
        plt.close()

# to cut the sample to silces
def cut_slice(data, begin, end):
    n = data[begin : end, :] # cut a slice of 100 data
    data = n.astype('float32') # numpy
    dataset = torch.from_numpy(data) # tensor
    # feed the dataset into a Datalosder
    # loaded_data = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataset, data

# ================================Start the Loop!========================================

# to record the bigining time of loop
start = time.time()

# =====================
# (1) difine variables
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
k = 0
index_J = []

# the parameters need to change
# beta1 = 0.1
# beta2 = 0.9
# batch_size = 50
# lambda_shift = 0.1

# =====================
# (2) creat the sample
# =====================

# load the sample, 2000 capacity data
sample = dataset

# ======================
# (3) training & testing
# ======================

# how many loops to try
while seed <= 1:

    loop[seed] = Loop(ORDER, DIM, LR, BATCH_SIZE, LAMBDA, index_G, index_D, BETA1, BETA2, seed)
    while j <= np.size(sample)//ORDER:
        # train the first silce
        if w_distance_s == {}:
            train_data, _ = cut_slice(sample, i - SLICE + 1, i )
            index_G, index_D, w_distance_s[j], w_costs = loop[seed].train_loop(EPOCH + k*50, CRITIC_ITERS, train_data, j)
            slice_data[j] = _
            k = k + 1

        # the W distance of the current slice does not meet the quality
        elif w_distance_s[j] >= THRESHOLD:
            train_data, _ = cut_slice(sample, i - SLICE + 1, i  )
            index_G, index_D, w_distance_s[j], w_costs = loop[seed].train_loop(EPOCH + k*50, CRITIC_ITERS, train_data, j)
            k = k + 1
        
        # use the trained netG and netD to test the data from next slice
        else:
            k = 0
            if i == np.size(sample)//ORDER:
                testing_data, dataset_slice = cut_slice(sample, j, i)
                w_distance_next = loop[seed].test(index_G, index_D, j, CRITIC_ITERS, train_data, testing_data)

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
                w_distance_next = loop[seed].test(index_G, index_D, j, CRITIC_ITERS, train_data, testing_data)
            

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
    # index_J = np.loadtxt('data/index_J.csv')
    
    # clear the w_distance_s for next loop
    w_distance_s = {}

    # to show the real data & fake data
    c = np.size(index_J)
    fig, axs = plt.subplots(c,1)
    plt.subplots_adjust(hspace=1)
    # fig.suptitle('Real & Fake Samples', weight='bold')
   
    
    axk = 0
    for ste in index_J:

        # lode the learned netG
        netG_fix = Generator(ORDER, DIM, ORDER)
        netG_fix.load_state_dict(torch.load(index_G[ste]))
        # netD_fix = Discriminator(ORDER, DIM, 1)
        # netD_fix.load_state_dict(torch.load(index_D[j])

        real_data = np.loadtxt('data' + '/' + str(seed) + '_'  +'slice' + '_' + str(int(ste)) + '.csv')
        # real_data = real_data[0:SLICE]
        real_dist = real_data.reshape(np.size(real_data), 1)

        noise = torch.randn(np.size(real_dist), ORDER)
        fake_data = netG_fix(noise)
        fake_dist = fake_data.data.numpy()

        n = np.append(real_dist, fake_dist, axis=1)
        data_fix = pd.DataFrame(data=n, columns=['Real data', 'Fake data'])


        num = ste + 1
        axs[axk].set_title('Slice %s' %num, weight='bold', loc='left', fontsize='small')
        axs[axk].set_xlabel('Iteration', labelpad=-8, fontsize='x-small')
        axs[axk].set_ylabel('Value', fontsize='x-small')
        sns.lineplot(data=data_fix, ax=axs[axk], lw=0.5)
        axs[axk].legend(loc='upper right', frameon=True, fontsize='xx-small')

        axk = axk + 1

    plt.savefig( 'ResultTry' + '_' + str(seed), dpi=300)


    # change parameters for next loop
    seed = seed + 1
    i = 1
    j = 0
    k = 0

    #if seed % 3 == 1:
    #    lambda_shift = lambda_shift - 0.01
    #    i = 1
    #    j = 0
    #    k = 0

    #else:
    #    i = 1
    #    j = 0
    #    k = 0

# record the ending time of the loop and print the time cost
end = time.time()
print('Runing time: %.8s s' % (end-start))
