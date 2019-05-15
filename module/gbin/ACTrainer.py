# -*- coding: utf-8 -*-
"""
__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

# from module.ACGAN import ACGAN
from module.betterAC import ACGAN
from module.datafunc import celeba_loader

import torch
from torchvision.utils import save_image

from tqdm import tqdm

class ACTrainer():
    def __init__(self, args):
        self.model = ACGAN(args) 

        self.G_optimizer = torch.optim.Adam(self.model.G.parameters())
        self.D_optimizer = torch.optim.Adam(self.model.D.parameters())
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

    def loss(self, conf, conf_t, label , label_t):
        return (self.criterion(conf, conf_t) + self.label_criterion(label, label_t))/2

    def train(self):
        for epoch in range(self.epochs):

            pbar = tqdm(self.dataloader, ncols=140)
            epoch_records = []
            count = 0
            pbar.set_description(str(epoch))

            for images, target in pbar:
                batch_size = images.shape[0]
                conf_real = torch.ones(batch_size).cuda()
                conf_fake = torch.zeros(batch_size).cuda()
                images = images.to(self.device)
                target = target.to(self.device)
                for __ in range(int(self.p)):
                    D_conf_real, D_lab_real = self.model.D(images)
                    D_loss_real = self.loss(D_conf_real, conf_real, D_lab_real, target)

                    z = torch.randn(batch_size, self.args.z_dim, device=self.device, requires_grad=True)
                    l = torch.LongTensor(batch_size).random_(0, self.args.n_classes).to(self.device)
                    # l.requires_grad = True

                    fake_images = self.model.G(z, l)
                    D_conf_fake, D_lab_fake = self.model.D(fake_images)
                    D_loss_fake = self.loss(D_conf_fake, conf_fake, D_lab_fake, l)

                    D_loss = (D_loss_real + D_loss_fake)/2
                    self.model.D.zero_grad()
                    D_loss.backward()
                    self.D_optimizer.step()

                    D_x = D_conf_real.mean().item()
                    D_acc_real = D_lab_real.argmax(dim=1).eq(target).sum()*100/len(target)
                    D_acc_fake = D_lab_fake.argmax(dim=1).eq(l).sum()*100/len(l)
                    D_acc = (D_acc_real+D_acc_fake)/2

                # Generator  maximize log(D(G(z)))
                for __ in range(int(self.q)):
                    
                    z = torch.randn(batch_size, self.args.z_dim, device=self.device, requires_grad=True)
                    l = torch.LongTensor(batch_size).random_(0, self.args.n_classes).to(self.device)

                    fake_images = self.model.G(z, l)

                    D_conf_fake, D_lab_fake = self.model.D(fake_images)
                    G_loss = self.loss(D_conf_fake, conf_real, D_lab_fake, l)
                    
                    self.G_optimizer.zero_grad()
                    G_loss.backward()
                    self.G_optimizer.step()
                    D_G_z = D_conf_fake.mean().item()

                if D_x > 0.8:
                    self.p = 1
                if D_x < 0.7:
                    self.p += 0.2
                else:
                    self.p = min(2, self.p)
                if D_x < 0.6: 
                    self.p += 0.2
                if D_x < 0.5:
                    self.p += 0.3
                if D_x < 0.4:
                    self.p += 0.3
                self.p = min(5, self.p)


                if D_G_z > 0.25:
                    self.q = min(3, self.q)
                if D_G_z > 0.3:
                    self.q = min(2, self.q)
                if D_G_z > 0.4:
                    self.q = 1

                if D_G_z < 0.2:
                    self.q += 0.1
                if D_G_z < 0.1:
                    self.q += 0.2
                if D_G_z < 0.05:
                    self.q += 0.3
                if D_G_z < 0.01:
                    self.q += 0.4
                if D_G_z < 0.005:
                    self.q += 0.5

                self.q = min(5, self.q)


                pbar.set_postfix(Dl=D_loss.item(), Gl=G_loss.item(), Dx=D_x, DGz=D_G_z, Dacc=D_acc.item(), p=int(self.p), q=int(self.q))
            self.model.save(self.modelpath)

            if (epoch+1) %10 ==0:
                self.save_one_sample_pair(self.img_dpath, epoch)
            if (epoch+1) % 50 == 0:
                self.model.save('%s.%d'%(self.modelpath, epoch))
            
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

 