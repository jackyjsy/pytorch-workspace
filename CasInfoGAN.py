from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import os
import itertools
from models import *
# from collections import OrderedDict
from torch.autograd import Variable
from collections import OrderedDict
# import itertools
# import util.util as util
# from util.image_pool import ImagePool
# from .base_model import BaseModel
# from . import networks
import sys
import util.util as util
from vgg16 import Vgg16



class CasInfoGAN():
    def _compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss
    def initialize(self, opt):
        self.opt = opt
        #self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        #self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        # self.nzc = opt.ncond + opt.hidden_size
        # self.Decoder = Decoder(self.nzc, opt.nc, opt.ngf)
        # self.Encoder = Encoder(self.nzc, opt.nc, opt.ngf)
        # self.netD = Discriminator(opt.nc, opt.ncond, opt.ndf)
        self.encoder = Encoder(opt)
        self.netG = generator(opt)
        self.netD = discriminator(opt)
        self.encoder.train()
        self.netG.train()
        self.netD.train()
        self.encoder.apply(weights_init)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        # self.Decoder.train()
        # self.Encoder.train()
        # self.netD.train()

        # self.Decoder.apply(weights_init)
        # self.Encoder.apply(weights_init)
        # self.netD.apply(weights_init)

        # # VGG feature
        # self.vgg = Vgg16()
        # # util.init_vgg16('./')
        # self.vgg.load_state_dict(torch.load(os.path.join('./', "vgg16.weight")))
        
        # load_state_dict
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.encoder, 'E', which_epoch)
            self.load_network(self.netG, 'G', which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', which_epoch)

        print(self.encoder)
        print(self.netG)
        print(self.netD)
        
        if self.opt.cuda:
            self.encoder.cuda()
            self.netG.cuda()
            self.netD.cuda()
            # self.vgg.cuda()

        if self.isTrain:
            if self.opt.cuda:
                self.BCE_loss = nn.BCELoss().cuda()
                self.CE_loss = nn.CrossEntropyLoss().cuda()
                self.MSE_loss = nn.MSELoss().cuda()
                self.cycle_loss_criterion = torch.nn.L1Loss().cuda()
            else:
                self.BCE_loss = nn.BCELoss()
                self.CE_loss = nn.CrossEntropyLoss()
                self.MSE_loss = nn.MSELoss()
                self.cycle_loss_criterion = torch.nn.L1Loss()
                
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.encoder.parameters(), self.netG.parameters()), lr=opt.lrG, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
            # self.optimizer_info = torch.optim.Adam(itertools.chain(self.netG.parameters(), self.netD.parameters()), lr=opt.lrD, betas=(opt.beta1, 0.999))

        
    def forward(self):
        self.x = Variable(self.x)
        self.zi = Variable(self.zi)
        self.cond_c = Variable(self.cond_c)
        self.cond_d = Variable(self.cond_d)
        self.label = Variable(self.label)

        

    def set_input(self, input_x, noise, continuous_cond, discrete_cond):
        if self.opt.cuda:
            # set input
            self.x = input_x.cuda()
            self.zi = noise.cuda()
            self.cond_c = continuous_cond.cuda()
            self.cond_d = discrete_cond.view(self.x.size(0),-1,1,1).cuda()
            # # cat together
            # self.zc = torch.cat([self.zi, cond_c, cond_d], 1)
            # label
            self.label = torch.FloatTensor(input_x.size(0),1,1,1)
            self.label = self.label.cuda()
        else:
            # set input
            self.x = input_x
            self.zi = noise
            self.cond_c = continuous_cond
            self.cond_d = discrete_cond
            # cat together
            self.zc = torch.cat([zi, cond_c, cond_d], 1)
            # label
            self.label = torch.FloatTensor(input_x.size(0),1,1,1)
            self.label = self.label

    # def backward_info(self):
    #     print(self.D_disc.size())
    #     print(self.cond_d.size())
    #     print(self.D_cont.size())
    #     print(self.cond_c.size())
    #     disc_loss = self.BCE_loss(self.D_disc, self.cond_d)
    #     cont_loss = self.MSE_loss(self.D_cont, self.cond_c)
    #     self.info_loss = disc_loss + cont_loss
    #     self.info_loss.backward()

    def backward_G(self):
        self.zi_fake = self.encoder(self.x)
        self.kl_loss = self._compute_kl(self.zi_fake)
        # update G network
        self.x_fake = self.netG(self.zi_fake, self.cond_c, self.cond_d)
        self.cycle_loss = self.cycle_loss_criterion(self.x_fake, self.x)
        self.D_fake = self.netD(self.x_fake, self.cond_d)
        self.label.data.fill_(1)
        self.loss_GAN = self.BCE_loss(self.D_fake, self.label)
        self.loss_G = self.loss_GAN + self.kl_loss + self.cycle_loss*10
        
        self.loss_G.backward(retain_graph=True)
        # self.loss_G.backward()
    def backward_D(self):
        # update D network
        self.D_real = self.netD(self.x, self.cond_d)
        
        # print(self.D_real)
        # print(x)
        # print(self.label)
        self.label.data.fill_(1)
        # print(self.label)
        D_real_loss = self.BCE_loss(self.D_real, self.label)
        D_real_loss.backward(retain_graph=True)
        D_real_loss.backward()
        self.zi_fake = self.encoder(self.x)
        x_fake = self.netG(self.zi_fake, self.cond_c, self.cond_d)
        # print(self.x)
        # print(x_fake)
        self.D_fake = self.netD(x_fake.detach(), self.cond_d)
        self.label.data.fill_(0)
        # print(self.label)
        D_fake_loss = self.BCE_loss(self.D_fake, self.label)
        # D_fake_loss.backward(retain_graph=True)
        D_fake_loss.backward()
        # print(x)
        self.D_loss = D_real_loss + D_fake_loss
        # self.D_loss.backward(retain_graph=True)

        
    def optimize_parameters(self):
        # forward
        self.forward()
        # D
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # # info
        # self.backward_info()
        # self.optimizer_info.step()

    def get_current_errors(self):
        # print(self.D_fake)
        # print(torch.mean(self.D_fake.data,0))
        # print(x)
        D_fake = torch.mean(self.D_fake.data,0).cpu().numpy()[0]
        D_real = torch.mean(self.D_real.data,0).cpu().numpy()[0]
        # D_fake = self.D_fake.data[0]
        # D_real = self.D_real.data[0]
        D_loss = self.D_loss.data[0]
        G_loss = self.loss_G.data[0]

        kl_loss = self.kl_loss.data[0]
        cycle_loss = self.cycle_loss.data[0]

        return OrderedDict([('D_fake', D_fake), ('D_real', D_real), ('D', D_loss), ('G', G_loss), ('kl_loss', kl_loss), ('cycle_loss', cycle_loss)])

    def get_current_visuals(self):
        real_x = util.tensor2im(self.x.data)
        fake_x = util.tensor2im(self.x_fake.data)

        return OrderedDict([('real_x', real_x), ('fake_x', fake_x)])
        
    def get_current_labels(self):
        c = self.cond_d
        c_fake = self.zc_fake[:,self.opt.hidden_size:,0,0]
        return c, c_fake

    def save(self, label):
        self.save_network(self.encoder, 'E', label)
        self.save_network(self.netG, 'G', label)
        self.save_network(self.netD, 'D', label)

    # def encode(self, image):
    #     image = image.cuda()
    #     self.input_image = Variable(image)
    #     print(image.shape)
    #     zc_fake = self.Encoder(self.input_image)
        
    #     # self.zc_fake = zc_fake.view(self.nzc,1)
    #     # print(zc_fake)
    #     noise_fake = zc_fake[:,:self.opt.hidden_size,:,:]
    #     condition_fake = zc_fake[:,self.opt.hidden_size:,:,:]
    #     # print(condition_fake)
    #     return noise_fake, condition_fake

    # def decode(self, noise, label):
    #     # if label is not None:
    #     #     zc_fake.data[self.opt.hidden_size:] = zc_fake.data[self.opt.hidden_size:] + label
        
    #     zc_fake = torch.cat((noise,label),1)
    #     self.image_generated = self.Decoder(zc_fake)
    #     return self.image_generated.data
        

    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    def vgg_loss(self, x_fake, x):
        real_vgg_feature = self.vgg(x)
        fake_vgg_feature = self.vgg(x_fake)

        real_vgg_feature_var1 = Variable(real_vgg_feature[0].data, requires_grad=False)
        real_vgg_feature_var2 = Variable(real_vgg_feature[1].data, requires_grad=False)
        real_vgg_feature_var3 = Variable(real_vgg_feature[2].data, requires_grad=False)
        real_vgg_feature_var4 = Variable(real_vgg_feature[3].data, requires_grad=False)
        

        loss_reg_1 = self.criterionVGG(fake_vgg_feature[0], real_vgg_feature_var1) 
        loss_reg_2 = self.criterionVGG(fake_vgg_feature[1], real_vgg_feature_var2) 
        loss_reg_3 = self.criterionVGG(fake_vgg_feature[2], real_vgg_feature_var3) 
        loss_reg_4 = self.criterionVGG(fake_vgg_feature[3], real_vgg_feature_var4)

        loss_reg = (loss_reg_1 + loss_reg_2 + loss_reg_3 + loss_reg_4)/64.0
        return loss_reg

