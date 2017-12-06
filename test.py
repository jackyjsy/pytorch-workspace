import time
from options.test_options import TestOptions
from data.data_loader import getLoader
from CasGAN import CasGAN
from util.visualizer import Visualizer
from util.util import read_and_transform_image as rtf
from util.util import tensor2im
import torch
from torch.autograd import Variable
opt = TestOptions().parse()
# data_loader = getLoader(opt)
# dataset_size = len(data_loader)
# print('#training images = %d' % dataset_size)

model = CasGAN()
model.initialize(opt)
# visualizer = Visualizer(opt)
# total_steps = 0
# fixed_noise = torch.FloatTensor(opt.valBatchSize, opt.hidden_size, 1, 1).uniform_(-1, 1)
# fixed_noise = fixed_noise.cuda()

image = rtf(opt.img_path,opt)
image = image.unsqueeze(0)
print(image.size())
n_fake, c_fake = model.encode(image)
vis = Visualizer(opt)

c_fake_m = c_fake.clone()
c_fake_m.data[0,0,0,0]=1
vis.plot_current_label((c_fake,c_fake_m),1)

img_generated = model.decode(n_fake,c_fake)
img_generated = tensor2im(img_generated)
from matplotlib import pyplot as plt
plt.imshow(img_generated, interpolation='nearest')
plt.show()
# img_generated = model.decode(n_fake,c_fake_m)
# img_generated = tensor2im(img_generated)
# from matplotlib import pyplot as plt
# plt.imshow(img_generated, interpolation='nearest')
# plt.show()

noise = Variable(torch.FloatTensor(1, opt.hidden_size, 1, 1).uniform_(-1, 1).cuda())
print(noise)
print(c_fake_m)
img_generated = model.decode(noise,c_fake_m)

img_generated = tensor2im(img_generated)
from matplotlib import pyplot as plt
plt.imshow(img_generated, interpolation='nearest')
plt.show()

