# -*- coding: utf-8 -*-
# Create figs

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import model
import pandas as pd

# define the theme of seaborn to drow images
sns.set_theme(palette="deep", style='ticks', color_codes=True, font='Times New Roman', font_scale=1)

class image:
    def __init__(self, order, dim, batch_size, epoch, seed):
        
        self.order = order
        self.dim = dim
        self.batch_size = batch_size
        self.epoch = epoch
        self.seed = seed

     # Generate and save a plot of the true distribution, the generator, and the cirtic(didcriminator)
    def creat_image(self, netG, real_data, index_image, w_loss):
        

        # use random normal noise to activate the netG
        noise = torch.randn(self.batch_size, self.order)
        # change tensor to numpy array 
        real_dist = real_data.numpy() 

        # generated fake samples
        fake_data = netG(noise)
        fake_dist = fake_data.data.numpy()

        # plot a 1 * 3 picture to show the generated data and the real data
        plt.clf()
        
        # plot an image with 3 axs
        fig, axs  = plt.subplots(1, 3, figsize=(15, 5))
        plt.subplots_adjust(wspace=0.2)
        # ax0 is the W loss
        axs[0].set_title('Wasserstein Distances', weight='bold')
        sns.lineplot(data=w_loss, ax=axs[0], estimator='mean', lw=2, color='steelblue')
        axs[0].set_xlabel('Iteration')
        axs[0].set_ylabel('Loss')
        
        # ax1 is the distributions of Real and Fake
        axs[1].set_title('Real & Fake Distributions', weight='bold')
        sns.kdeplot(real_dist.flatten(), fill=True, ax=axs[1], label='Real data', color='b', bw_adjust=4)
        sns.kdeplot(fake_dist.flatten(), fill=True, ax=axs[1], label='Fake data', color='r', bw_adjust=4)
        axs[1].set_xlabel('Sample Value')
        axs[1].set_ylabel('Probability density')
        axs[1].legend(loc='upper right', frameon=True)
        
        # ax2 is the samples of Real and Fake
        axs[2].set_title('Real & Fake Samples', weight='bold')
        sns.histplot(x=real_dist.flatten(), y=fake_dist.flatten(), cmap='rocket_r', cbar=True)
        sns.kdeplot(x=real_dist.flatten(), y=fake_dist.flatten(), ax=axs[2])
        axs[2].set_xlabel('Real Samples')
        axs[2].set_ylabel('Fake Samples')

        plt.savefig('images' + '/' + '_' + str(self.seed) + '_' + str(index_image), dpi=300)
        plt.close()
        

    # drow a fig to show the real data & fake data
    def real_fake_image(self, index_J):
            
        c = np.size(index_J)
        fig, axs = plt.subplots(c,1, figsize=(10,30))
        fig.subplots_adjust(hspace=2)
        # fig.suptitle('Real & Fake Samples', weight='bold')

        indexG = {}
        indexD = {}
    
        axk = 0
        for ste in index_J:
            
            # initial the shapes of netG and netD
            netG_l = model.Generator(self.order, self.dim, self.order)
            netD_l = model.Discriminator(self.order, self.dim, self.order)

            # pump out the index of tthe state dict of netG and ndtD
            indexG[ste] = 'models' + '/' + 'modelG' + '_' + str(self.order) + '_' + str(int(ste))
            indexD[ste] = 'models' + '/' + 'modelD' + '_' + str(self.order) + '_' + str(int(ste))

            # load the weights of netG and netD
            netG_l.load_state_dict(torch.load(indexG[ste]))
            netG_l.eval()
            netD_l.load_state_dict(torch.load(indexD[ste]))
            netD_l.eval()


            real_data = np.loadtxt('data' + '/' + str(self.seed) + '_'  +'slice' + '_' + str(int(ste)) + '.csv')
            real_dist = real_data.reshape(np.size(real_data), 1)

            noise = torch.randn(np.size(real_dist), self.order)
            fake_data = netG_l(noise)
            fake_dist = fake_data.data.numpy()

            n = np.append(real_dist, fake_dist, axis=1)
            data_fix = pd.DataFrame(data=n, columns=['Real data', 'Fake data'])


            num = ste + 1
            axs[axk].set_title('Slice %s' %num, weight='bold', loc='left', fontsize='small')
            axs[axk].set_xlabel('Iteration', labelpad=-1, fontsize='x-small')
            axs[axk].set_ylabel('Value', fontsize='x-small')
            sns.lineplot(data=data_fix, ax=axs[axk], lw=0.5)
            axs[axk].legend(loc='upper right', frameon=True, fontsize='xx-small')

            axk = axk + 1
        plt.savefig('Result' + '_' + str(self.seed), dpi=300)
        plt.close()
