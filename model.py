
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import autograd
import matplotlib.pyplot as plt
import seaborn as sns
from Utils import image_processing

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
    def __init__(self, order, dim, lr, batch_size, lambda_, epoch, threshold,
                 indexG, indexD, critic_iters, cut,  beta1, beta2, seed):

        self.order = order
        self.dim = dim
        self.Lr = lr
        self.batch_size = batch_size
        self.lamba = lambda_
        self.epoch = epoch
        self.critic_iters = critic_iters
        self.cut = cut
        self.threshold = threshold
    
        # the list that save the pass of nets' weights, NOT net itself
        self.indexG = indexG
        self.indexD = indexD

        # the BETAs of Adam optimization
        self.beta1 = beta1
        self.beta2 = beta2

        self.seed = seed


    # the training process of netG and netD
    def train_loop(self, training_data, index):

        
        # the sliced data
        training_data = DataLoader(training_data, batch_size=self.batch_size, shuffle=True)


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
        for iteration in range(self.epoch):
            
            # =================
            # (1) Update netD
            # =================

            # free the gradient of netD
            for p in self.netD.parameters(): 
                p.requires_grad = True

            # the iteration of netD tring
            for iter_d in range(self.critic_iters):
                
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
            
            
        # save the state dict(weights) of current netG and netD
        torch.save(self.netG.state_dict(), 'models' + '/' + 'modelG' + '_' + str(self.seed) + '_' +str(index))
        torch.save(self.netD.state_dict(), 'models' + '/' + 'modelD' + '_' + str(self.seed) + '_' +str(index))

        # pump out the index of tthe state dict of netG and ndtD
        self.indexG[index] = 'models' + '/' + 'modelG' + '_' + str(self.seed) + '_' + str(index)
        self.indexD[index] = 'models' + '/' + 'modelD' + '_' + str(self.seed) + '_' + str(index)
        indexG = self.indexG
        indexD = self.indexD
        
        # save an image
        image = image_processing.image(self.order, self.dim, self.batch_size, self.epoch, self.seed)
        image.creat_image(self.netG, _data, index, w_cost)

        # calculate the mean of last 5 w distence for output
        print(w_cost[-self.cut:])
        w_dist = np.mean(w_cost[-self.cut:])
        print('W distance: ' + str(w_dist))
        
        return indexG, indexD, w_dist, w_cost
        
    # the test loop to identify the switch of distribution
    def test(self, indexG_in, indexD_in, index_in, test_data_in):

        index = index_in
        indexG = indexG_in
        indexD = indexD_in

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

        for i in range(self.epoch):

            for iter_d in range(self.critic_iters):
                _data = next(iter(test_data))
                real_data = _data
                
                self.netD.zero_grad()
                
                real_D = self.netD(real_data)
                real_D = real_D.mean()

                real_cost = real_D
                real_cost.backward(m_one)
                optimizerD.step()
            
            new_D = real_cost.detach().numpy()
            new_dist.append(new_D)
            cr = self.critic_iters // 2
            mean_new = np.std(new_dist[-cr:])
            
            if i > self.cut and mean_new < self.threshold:
                break
        
        for p in self.netD.parameters(): 
                p.requires_grad = False
        
        # calculate the W distance 5 times and return the mean
        ii = 1
        w_dist = []
        while ii <= self.critic_iters:

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

