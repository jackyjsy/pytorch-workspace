import torch
from PIL import Image
import os
import os.path
import numpy as np
import scipy.io
def getLoader(opt):
    dataroot = opt.dataroot
    originalSize = opt.loadSize
    imageSize = opt.fineSize
    batchSize = opt.batchSize
    workers = 4
    mean=(0.5, 0.5, 0.5)
    std=(0.5, 0.5, 0.5)
    split='train'
    shuffle=True
    seed=None
    if opt.dataset == 'LFWA':
        from data.folder import LFWA_Dataset as commonDataset
        import torchvision.transforms as transforms
        dataset = commonDataset(root=dataroot,
                                transform=transforms.Compose([
                                transforms.Scale(imageSize),
                                transforms.RandomCrop(imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std),
                                ]),
                                seed=seed)
    elif opt.dataset == 'CelebA':
        print(opt.dataset)
        from data.folder import ImageFolder as commonDataset
        import torchvision.transforms as transforms
        dataset = commonDataset(root=dataroot,
                                transform=transforms.Compose([
                                transforms.Scale(imageSize),
                                transforms.RandomCrop(imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std),
                                ]),
                                seed=seed)
    else:
        print('No matched dataset found, check data_loader.py.')
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=batchSize, 
                                            shuffle=shuffle, 
                                            num_workers=int(workers))


    return dataloader


