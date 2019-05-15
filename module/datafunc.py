# -*- coding: utf-8 -*-
"""
__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

import os
import csv
from PIL import Image

import torch
import torchvision

class CELEBA(torch.utils.data.Dataset):
    """docstring for CELEBA"""
    def __init__(self):
        super(CELEBA, self).__init__()
        self.labelpath = 'hw3_data/face/train.csv'
        self.imgdir = 'hw3_data/face/train'
        self.length = len(os.listdir(self.imgdir))
        self.str_ =  lambda i: '0' * (5 - len(str(i))) + str(i)

        self._readlabel()
        self.trans = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        imgfile = '%s.png'%self.str_(index)
        label = self.labeldict[imgfile]
        label = torch.LongTensor([label]).squeeze()
        
        imgpath = os.path.join(self.imgdir, imgfile)
        img = Image.open(imgpath)
        img = self.trans(img)

        return img, label

    def _readlabel(self):
        self.labeldict = {}
        with open(self.labelpath, newline='') as f:
            reader = csv.reader(f)
            # next(reader, None)
            first = True
            for row in reader:
                if first:
                    key = row.index('Smiling')
                    first = False
                else:
                    self.labeldict[row[0]]=int(float(row[key]))


# def CELEBA(data_dir='hw3_data/face'):
#     trans = torchvision.transforms.Compose([
#         torchvision.transforms.ToTensor(),
#         ])

#     dataset = torchvision.datasets.ImageFolder(data_dir,trans)
#     return dataset

def celeba_loader(batch_size=128):
    dataset = CELEBA()
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == '__main__':
    dataloader = celeba_loader()
    for image, label in dataloader:
        print(image.shape)
        print(image[0,:2,:2,:2])
        print(label.shape)
        print(label)
        break
