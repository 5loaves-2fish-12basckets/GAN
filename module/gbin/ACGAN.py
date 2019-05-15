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

        self.init_size = args.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(args.z_dim, 128 * self.init_size ** 2)) # first enlarge with DNN

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            Upsample(scale_factor=2),
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            Upsample(scale_factor=2),
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, args.img_channel_num, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)    # B, 100
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(args.img_channel_num, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = args.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, args.n_classes), nn.Softmax(dim=1))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
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

'''
target for D confidence is same as output (N,) --> use BCE loss
target for D label is (N,), output for D label is (N, class) --> use CrossEntropyLoss
'''
