# -*- coding: utf-8 -*-
"""
__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

from module.WACGAN import ACGAN
from module.datafunc import celeba_loader

import torch
from torchvision.utils import save_image

from tqdm import tqdm

class ACTrainer():
    def __init__(self, args):
        self.model = ACGAN() 

        self.G_optimizer = torch.optim.Adam(self.model.G.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.D_optimizer = torch.optim.Adam(self.model.D.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.dataloader = celeba_loader(args.batch_size)

        self.epochs = args.epochs
        self.modelpath = 'ckpt/%s/model.pth'%args.taskname
        self.img_dpath = 'ckpt/'+args.taskname+'/img_%d-%s.png'
        self.args = args
        if args.resume is not None:
            print('resuming model', args.resume)
            self.model.load(args.resume)
            

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        self.LAMBDA = 10
        self.p = 1
        self.q = 1
        self.one = torch.FloatTensor([1]).to(self.device)
        self.mone = self.one * -1
                    
        print('finish init')


    def train(self):
        gen_iterations = 0
        for epoch in range(self.epochs):

            pbar = tqdm(range(len(self.dataloader)), ncols=110)
            pbar.set_description(str(epoch))

            # indicator = 0
            # if gen_iterations < 25 or gen_iterations % 500 == 0:
                # self.p = 10
            # else: 
            self.p = 1

            for i in pbar:

                # indicator += 1

                for p in self.model.D.parameters():
                    p.requires_grad = True
                    # p.data.clamp_(-0.07, 0.07)

                images, target = next(iter(self.dataloader))
                images, target = images.to(self.device), target.to(self.device)

                self.model.D.zero_grad()
                batch_size = images.shape[0]
                D_conf_real, D_lab_real = self.model.D(images)
                D_conf_real.mean().backward(self.mone, retain_graph=True)
                Dlloss_real = self.criterion(D_lab_real, target)
                Dlloss_real.backward()

                z = torch.randn(batch_size, self.args.z_dim - 1, device=self.device, requires_grad=True)
                l = torch.LongTensor(batch_size).random_(0, self.args.n_classes).to(self.device)

                fake_images = self.model.G(z, l)
                D_conf_fake, D_lab_fake = self.model.D(fake_images)
                D_conf_fake.mean().backward(self.one, retain_graph=True)
                Dlloss_fake = self.criterion(D_lab_fake, l)
                Dlloss_fake.backward()

                # penalty = self.calculate_gradient(batch_size, images.data, fake_images.data)
                # penalty.backward()

                Dlloss = (Dlloss_fake + Dlloss_real)/2
                D_loss = (D_conf_fake - D_conf_real).mean().item()
                # D_loss = (D_conf_fake - D_conf_real + penalty).mean().item()
                Wasserstein_D = (D_conf_real - D_conf_fake).mean().item()

                self.D_optimizer.step()

                D_acc_real = D_lab_real.argmax(dim=1).eq(target).sum()*100/len(target)
                D_acc_fake = D_lab_fake.argmax(dim=1).eq(l).sum()*100/len(l)
                D_acc = (D_acc_real+D_acc_fake)/2
                # pbar.set_postfix(Dacc=D_acc.item(), Dl=Dlloss.item(), DW = Wasserstein_D, D_loss = D_loss)

                # if indicator == self.p:

                    # indicator = 0 # train once
                    # Generator  maximize log(D(G(z)))
                for p in self.model.D.parameters():
                    p.requires_grad = False

                self.model.G.zero_grad()

                z = torch.randn(batch_size, self.args.z_dim - 1, device=self.device, requires_grad=True)
                l = torch.LongTensor(batch_size).random_(0, self.args.n_classes).to(self.device)

                fake_images = self.model.G(z, l)

                D_conf_fake, D_lab_fake = self.model.D(fake_images)
                D_conf_fake.mean().backward(self.mone, retain_graph=True)
                Glloss = self.criterion(D_lab_fake, l)
                Glloss.backward()

                self.G_optimizer.step()
                gen_iterations += 1

                pbar.set_postfix(Dacc=D_acc.item(), Dl=Dlloss.item(), Gl=Glloss.item(), DW = Wasserstein_D, D_loss = D_loss)
                self.save_one_sample_pair(self.img_dpath, epoch)
            self.model.save(self.modelpath)

    def save_one_sample_pair(self, img_name='generated.png', epoch=0):
        with torch.no_grad():
            z = torch.randn(10, self.args.z_dim - 1, device=self.device, requires_grad=True)
            l = torch.LongTensor(10).fill_(0).to(self.device)
            l2 = torch.LongTensor(10).fill_(1).to(self.device)

            generated_img = self.model.G(z, l)
            generated_img2 = self.model.G(z, l2)
            save_image(generated_img.cpu(), img_name%(epoch, 'n'))
            save_image(generated_img2.cpu(), img_name%(epoch, 'y'))

 