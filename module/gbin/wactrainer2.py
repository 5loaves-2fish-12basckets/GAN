# -*- coding: utf-8 -*-
"""
__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

from module.betterAC import ACGAN
from module.datafunc import celeba_loader

import torch
from torchvision.utils import save_image

from tqdm import tqdm

class ACTrainer():
    def __init__(self, args):
        self.model = ACGAN(args) 

        self.G_optimizer = torch.optim.Adam(self.model.G.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.D_optimizer = torch.optim.Adam(self.model.D.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.criterion = torch.nn.BCELoss()
        self.label_criterion = torch.nn.CrossEntropyLoss()
        self.dataloader = celeba_loader(args.batch_size)

        self.epochs = args.epochs
        self.modelpath = 'ckpt/%s/model.pth'%args.taskname
        self.img_dpath = 'ckpt/'+args.taskname+'/img_%d-%s.png'
        self.args = args
        if args.resume is not None:
            print('loading model')
            self.model.load(self.modelpath)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        

        self.p = 1
        self.q = 1
        print('finish init')

    # def loss(self, conf, conf_t, label , label_t):
    #     return (self.criterion(conf, conf_t) + self.label_criterion(label, label_t))/2

    def calc_gradient_penalty(self, real_images, fake_images):
        

    def train(self):
        gen_iterations = 0
        for epoch in range(self.epochs):

            pbar = tqdm(range(len(self.dataloader)), ncols=110)
            pbar.set_description(str(epoch))

            for i in pbar:
                for p in self.model.D.parameters():
                    p.requires_grad = True

                if gen_iterations < 25 or gen_iterations % 500 == 0:
                    self.p = 100
                    holder = tqdm(range(self.p), ncols=100)

                else: 
                    self.p = 5
                    holder = range(self.p)

                for j in holder:
                    # if self.p > 5:
                    for p in self.model.D.parameters():
                        p.data.clamp_(-0.03, 0.03)
                    # else:
                    #     for p in self.model.D.parameters():
                    #         p.data.clamp_(-0.3, 0.3)

                    images, target = next(iter(self.dataloader))
                    images, target = images.to(self.device), target.to(self.device)

                    self.model.D.zero_grad()
                    batch_size = images.shape[0]
                    self.one = torch.FloatTensor([1 for __ in range(batch_size)]).to(self.device)
                    self.mone = self.one * -1
                    D_conf_real, D_lab_real = self.model.D(images)
                    D_conf_real.backward(self.one, retain_graph=True)
                    Dlloss_real = self.label_criterion(D_lab_real, target)
                    Dlloss_real.backward()

                    z = torch.randn(batch_size, self.args.z_dim, device=self.device, requires_grad=True)
                    l = torch.LongTensor(batch_size).random_(0, self.args.n_classes).to(self.device)

                    fake_images = self.model.G(z, l)
                    D_conf_fake, D_lab_fake = self.model.D(fake_images)
                    D_conf_fake.backward(self.mone, retain_graph=True)
                    Dlloss_fake = self.label_criterion(D_lab_fake, l)
                    Dlloss_fake.backward()

                    Dlloss = (Dlloss_fake + Dlloss_real)/2
                    self.D_optimizer.step()

                    D_x = D_conf_real.mean().item()

                    D_acc_real = D_lab_real.argmax(dim=1).eq(target).sum()*100/len(target)
                    D_acc_fake = D_lab_fake.argmax(dim=1).eq(l).sum()*100/len(l)
                    D_acc = (D_acc_real+D_acc_fake)/2

                # Generator  maximize log(D(G(z)))
                for p in self.model.D.parameters():
                    p.requires_grad = False

                self.model.G.zero_grad()

                z = torch.randn(batch_size, self.args.z_dim, device=self.device, requires_grad=True)
                l = torch.LongTensor(batch_size).random_(0, self.args.n_classes).to(self.device)

                fake_images = self.model.G(z, l)

                D_conf_fake, D_lab_fake = self.model.D(fake_images)
                D_conf_fake.backward(self.one, retain_graph=True)
                Glloss = self.label_criterion(D_lab_fake, l)
                Glloss.backward()

                self.G_optimizer.step()
                gen_iterations += 1

                D_G_z = D_conf_fake.mean().item()
                pbar.set_postfix(Dl=Dlloss.item(), Gl=Glloss.item(), Dx=D_x, DGz=D_G_z, Dacc=D_acc.item())


                # if (epoch+1) %10 ==0:
                
                self.save_one_sample_pair(self.img_dpath, 0)

                if gen_iterations % 500 == 0:
                    self.save_one_sample_pair(self.img_dpath, gen_iterations)
                    self.model.save('%s.%d'%(self.modelpath, gen_iterations))
            
        self.model.save(self.modelpath)

    def save_one_sample_pair(self, img_name='generated.png', epoch=0):
        with torch.no_grad():
            z = torch.randn(10, self.args.z_dim, device=self.device, requires_grad=True)
            l = torch.LongTensor(10).fill_(0).to(self.device)
            l2 = torch.LongTensor(10).fill_(1).to(self.device)

            generated_img = self.model.G(z, l)
            generated_img2 = self.model.G(z, l2)
            save_image(generated_img.cpu(), img_name%(epoch, 'n'))
            save_image(generated_img2.cpu(), img_name%(epoch, 'y'))

 