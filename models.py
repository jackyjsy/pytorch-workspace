import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            #m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            #m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, opt):
        dataset = opt.dataset
        super(generator, self).__init__()
        if dataset == 'LFWA':
            self.input_height = 64
            self.input_width = 64
            
            self.noise_dim = opt.hidden_size
            self.n_cat = [1,3,5,4,3]
            self.n_disrete_code = len(self.n_cat) # 5
            self.len_discrete_code = sum(self.n_cat)  # categorical distribution (i.e. label) #16
            self.len_continuous_code = 10  # gaussian distribution (e.g. rotation, thickness)
            
            self.input_dim = self.noise_dim + self.len_discrete_code + self.len_continuous_code
            self.output_dim = 3

        if dataset == 'CelebA':
            self.input_height = 128
            self.input_width = 128
            
            self.noise_dim = opt.hidden_size
            self.n_cat = [1,1,5,1,1,1]
            self.n_disrete_code = len(self.n_cat) # 5
            self.len_discrete_code = sum(self.n_cat)  # categorical distribution (i.e. label) #16
            self.len_continuous_code = 10  # gaussian distribution (e.g. rotation, thickness)
            
            self.input_dim = self.noise_dim + self.len_discrete_code + self.len_continuous_code
            self.output_dim = 3

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 2*2*1024),
            nn.BatchNorm1d(2*2*1024),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            # 2x2
            nn.ConvTranspose2d(1024, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            # 4x4
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 8x8
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 16x16
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64x64
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
            # 128x128
        )
        # initialize_weights(self)

    def forward(self, input, cont_code, dist_code):
        # print(input.size())
        # print(cont_code.size())
        # print(dist_code.size())
        x = torch.cat([input, cont_code, dist_code], 1)
        # print(x.size())
        # print(self.fc)
        # print(x)
        x = x.view(input.size(0),-1)
        # print(x)
        x = self.fc(x)
        x = x.view(-1, 1024, 2, 2)
        x = self.deconv(x)
        return x

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, opt):
        super(discriminator, self).__init__()
        dataset = opt.dataset
        print(dataset)
        if dataset == 'LFWA':
            self.input_height = 64
            self.input_width = 64
            self.input_dim = 3
            self.output_dim = 1
            self.n_cat = [1,3,5,4,3]
            self.n_disrete_code = len(self.n_cat) # 5
            self.len_discrete_code = sum(self.n_cat)  # categorical distribution (i.e. label) #16
            self.len_continuous_code = 10  # gaussian distribution (e.g. rotation, thickness)
        if dataset == 'CelebA':
            self.input_height = 128
            self.input_width = 128
            self.input_dim = 3
            self.output_dim = 1
            self.n_cat = [1,1,5,1,1,1]
            self.n_disrete_code = len(self.n_cat) # 5
            self.len_discrete_code = sum(self.n_cat)  # categorical distribution (i.e. label) #16
            self.len_continuous_code = 10  # gaussian distribution (e.g. rotation, thickness)
        
        cond_size = sum(self.n_cat)
        # elif dataset == 'CelebA':
        self.conv = nn.Sequential(
            # 64x64
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            # 32x32
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # 16x16
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # 8x8
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # 4x4x512
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            # 4x4x512
            nn.Conv2d(1024, 1024, 4, 1, 0),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(512 * 4 * 4, self.output_dim + self.len_continuous_code + self.len_discrete_code),
        #     # nn.Sigmoid(),
        # )
        # )
        self.fc = nn.Sequential(
            nn.Linear(1024 + cond_size, self.output_dim),
            nn.Sigmoid(),
        )
        # initialize_weights(self)

    def forward(self, input, cond):
        # print(input)
        x = self.conv(input)
        # print(x)
        # x = x.view(-1, 512 * 4 * 4)
        # print(x)
        # print(x.size())
        x = torch.cat([x, cond], 1)
        # print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        # # print(x[:, 0])
        # a = F.sigmoid(x[:, 0].clone())
        # # print(a)
        # # print(a)
        # b = x[:, self.output_dim:self.output_dim + self.len_continuous_code].clone()
        # start = self.output_dim + self.len_continuous_code
        # c = x[:, self.output_dim + self.len_continuous_code:].clone()
        # for n_each_cat in self.n_cat:
        #     end = start + n_each_cat
        #     c[:, start - self.output_dim - self.len_continuous_code:end - self.output_dim - self.len_continuous_code] = F.softmax(x[:, start:end])
        #     start = end
        # return a, b, c

class Decoder(nn.Module):
    def __init__(self, nzc, nc, ngf, ngpu=1):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        # input is zc, going into a convolution
        self.main = nn.Sequential(nn.ConvTranspose2d(nzc, ngf * 16, 4, 1, 0, bias=False),
                                nn.BatchNorm2d(ngf * 16),
                                nn.ReLU(True),
                                # state size. N x (ngf*16) x 4 x 4
                                nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
                                nn.BatchNorm2d(ngf * 8),
                                nn.ReLU(True),
                                # state size. (ngf*4) x 8 x 8
                                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                                nn.BatchNorm2d(ngf * 4),
                                nn.ReLU(True),
                                # state size. (ngf*2) x 16 x 16
                                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                                nn.BatchNorm2d(ngf * 2),
                                nn.ReLU(True),
                                # state size. (ngf*2) x 32 x 32
                                nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                                nn.BatchNorm2d(ngf),
                                nn.ReLU(True),
                                # state size. (ngf) x 64 x 64
                                nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
                                nn.Tanh())
                                # state size. (nc) x 128 x 128
    def forward(self, zc):
        x = self.main(zc)
        return x

class GaussianNoiseLayer(nn.Module):
    def __init__(self,):
        super(GaussianNoiseLayer, self).__init__()

    def forward(self, x):
        if self.training == False:
            return x
        noise = Variable(torch.randn(x.size()).cuda(x.data.get_device()))
        return x + noise/10

class Encoder(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, opt):
        super(Encoder, self).__init__()
        dataset = opt.dataset
        print(dataset)
        if dataset == 'LFWA':
            self.input_height = 64
            self.input_width = 64
            self.input_dim = 3
            self.output_dim = 512

        if dataset == 'CelebA':
            self.input_height = 128
            self.input_width = 128
            self.input_dim = 3
            self.output_dim = 512

        # elif dataset == 'CelebA':
        self.conv = nn.Sequential(
            # 64x64
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            # 32x32
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # 16x16
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # 8x8
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # # 4x4x512
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024 * 4 * 4, self.output_dim),
            GaussianNoiseLayer(),
            # nn.Sigmoid(),
        )
        # initialize_weights(self)

    def forward(self, input):
        # print(input)
        x = self.conv(input)
        # print(x)
        x = x.view(-1, 1024 * 4 * 4)
        # print(x)
        # print(x.size())
        x = self.fc(x)
        
        return x
