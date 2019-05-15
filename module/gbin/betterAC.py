# -*- coding: utf-8 -*-
"""Function for DCGAN model class delaration
__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

import torch
import torch.nn as nn


class ACGAN(nn.Module):
    def __init__(self, args):
        super(ACGAN, self).__init__()
        self.G = Generator(args)    # noise (B, 128) label (B, 1)
        self.G.apply(weights_init)
        self.D = Discriminator(args)  # img
        self.D.apply(weights_init)

    def save(self, filepath):
        state = {
            'gen_net': self.G.state_dict(),
            'dis_net': self.D.state_dict(),
        }
        torch.save(state, filepath)

    def load(self, filepath):
        state = torch.load(filepath)
        self.G.load_state_dict(state['gen_net'])
        self.D.load_state_dict(state['dis_net'])


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class Upsample(nn.Module):
    def __init__(self, scale_factor=2):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor)


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(args.n_classes, args.z_dim)

        self.img_channel_num = 3
        self.main = nn.ModuleList()
        self.ngf = 128
        self.layer_G = [(self.ngf*8,4,1,0), (self.ngf*4,4,2,1), (self.ngf*2,4,2,1), (self.ngf,4,2,1), (self.img_channel_num,4,2,1)]

        in_channel = args.z_dim
        for i, x in enumerate(self.layer_G):
            self.main.append(nn.ConvTranspose2d(in_channel, *x))
            in_channel = x[0]
            if i < len(self.layer_G)-1:
                self.main.append(nn.BatchNorm2d(in_channel))
                self.main.append(nn.ReLU())
            else:
                self.main.append(nn.Tanh())


    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        x = gen_input.view(gen_input.size(0), gen_input.size(1), 1,1)

        for layer in self.main:
            x = layer(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.main = nn.ModuleList()
        self.img_channel_num = 3
        self.ndf = 128
        self.layer_D = [(self.ndf,4,2,1), (self.ndf*4,4,2,1), (self.ndf*8,4,2,1), (self.ndf,4,2,1),]# (1,4,1,0)]

        in_channel = args.img_channel_num
        for i, x in enumerate(self.layer_D):
            self.main.append(nn.Conv2d(in_channel, *x))
            in_channel = x[0]
            if i > 0:
                self.main.append(nn.BatchNorm2d(in_channel))
            self.main.append(nn.LeakyReLU(0.2))
            self.main.append(nn.Dropout2d(0.25))

        self.adv_layer = nn.Sequential(nn.Linear(128 * 4 ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * 4 ** 2, args.n_classes), nn.Softmax(dim=1))

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        out = x.view(x.shape[0], -1)
        validity = self.adv_layer(out).squeeze()
        label = self.aux_layer(out)

        return validity, label

if __name__ == '__main__':
    class model_param():
        def __init__(self):
            self.img_channel_num = 3
            self.z_dim = 100
            self.n_classes = 2
            self.img_size = 64
    args = model_param()
    model = ACGAN(args)
    print(model)
    batch_size = 100
    z = torch.randn(batch_size, args.z_dim, requires_grad=True)
    l = torch.LongTensor(batch_size).random_(0, args.n_classes)
    img = model.G(z, l)
    print(img.shape)

    sample = torch.rand(13, 3,64,64)
    a,b = model.D(sample)
    print(a.shape)
    print(b.shape)
    print(a)
    print(b)

'''
target for D confidence is same as output (N,) --> use BCE loss
target for D label is (N,), output for D label is (N, class) --> use CrossEntropyLoss
        
'''
