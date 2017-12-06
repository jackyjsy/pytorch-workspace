import time
from options.train_options import TrainOptions
from data.data_loader import getLoader
from CasInfoGAN import CasInfoGAN
from util.visualizer import Visualizer
import torch

opt = TrainOptions().parse()
data_loader = getLoader(opt)
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = CasInfoGAN()
model.initialize(opt)
visualizer = Visualizer(opt)
total_steps = 0
# fixed_noise = torch.FloatTensor(opt.valBatchSize, opt.hidden_size, 1, 1).uniform_(-1, 1)
# fixed_noise = fixed_noise.cuda()
n_cond_c = 10

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
# for epoch in range(1):
    print('epoch %d started!' % epoch)
    epoch_start_time = time.time()
    epoch_iter = 0
    for i, data in enumerate(data_loader, 0):
        # if i < dataset_size/2:
        #     continue
        iter_start_time = time.time()

        x, cond_d = data
        # print(x.size())
        # batchSize = x.size(0)
        batchSize = opt.batchSize
        total_steps += batchSize
        epoch_iter += batchSize
        # noise = torch.FloatTensor(x.size(0), opt.hidden_size, 1, 1).uniform_(-1, 1)
        noise = torch.FloatTensor(x.size(0), opt.hidden_size, 1, 1).normal_(0, 1)
        cond_c = torch.FloatTensor(x.size(0), n_cond_c, 1, 1).uniform_(-1, 1)

        # print('setting input data to model!')
        model.set_input(x, noise, cond_c, cond_d)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)
            # visualizer.plot_current_label(model.get_current_labels(), epoch)
            # visualizer.display_current_results(model.get_current_visuals(), epoch_iter)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/(dataset_size * opt.batchSize), opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
################################################################
    # if epoch > opt.niter:
    #     model.update_learning_rate()
