# -*- coding: utf-8 -*-
"""
__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

from module.DCGAN import DCGAN
from module.datafunc import celeba_loader
import torch
from torchvision.utils import save_image

from tqdm import tqdm


class Trainer():
    def __init__(self, args):
        self.model = DCGAN() 
        self.G_optimizer = torch.optim.Adam(self.model.G.parameters())
        self.D_optimizer = torch.optim.Adam(self.model.D.parameters())
        self.criterion = torch.nn.BCELoss()
        self.dataloader = celeba_loader(args.batch_size)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.k = 1

        self.epochs = args.epochs
        self.modelpath = 'ckpt/%s/model.pth'%args.taskname
        self.img_dpath = 'ckpt/'+args.taskname+'/img_%d.png'
    def train(self):
        for epoch in range(self.epochs):
            real_label = 1
            fake_label = 0

            pbar = tqdm(self.dataloader, ncols=100)
            epoch_records = []
            count = 0
            pbar.set_description(str(epoch))

            for images, __ in pbar:
                batch_size = images.shape[0]
                label_real = torch.ones(batch_size).cuda()
                label_fake = torch.zeros(batch_size).cuda()
                images = images.to(self.device)
                for __ in range(1):
                    D_out_real = self.model.D(images).view(-1)
                    D_x = D_out_real.mean().item()
                    
                    # print(D_out_real)
                    # print(label_real)
                    err_D_real = self.criterion(D_out_real, label_real)

                    z = torch.randn(batch_size, 100, 1, 1, device=self.device, requires_grad=True)

                    fake_images = self.model.G(z)
                    D_out_fake = self.model.D(fake_images).view(-1)
                    
                    err_D_fake = self.criterion(D_out_fake, label_fake)

                    D_loss = (err_D_fake + err_D_real)/2
                    self.model.D.zero_grad()
                    D_loss.backward()
                    self.D_optimizer.step()

                # Generator  maximize log(D(G(z)))
                for __ in range(self.k):
                    
                    z = torch.randn(batch_size, 100, 1, 1, device=self.device, requires_grad=True)
                    fake_images = self.model.G(z)

                    D_out_fake = self.model.D(fake_images).view(-1)

                    D_G_z = D_out_fake.mean().item()

                    G_loss = self.criterion(D_out_fake, label_real)
                    
                    self.model.D.zero_grad()
                    self.model.G.zero_grad()
                    G_loss.backward()
                    self.G_optimizer.step()

                # if D_G_z < 0.01:
                #     self.k += 1
                #     self.k = min(self.k, 5)                        
                # elif D_G_z > 0.2:
                #     self.k -= 1
                #     self.k = max(self.k, 1)
                # if D_G_z > 0.4:
                #     self.k = 1
                pbar.set_postfix(Dloss=D_loss.item(), Gloss=G_loss.item(), D_x=D_x, D_G_z=D_G_z, k=self.k)
            
            self.model.save(self.modelpath)

            if (epoch+1) %10 ==0:
                self.model.save('%s.%d'%(self.modelpath, epoch))
                self.save_one_sample(self.img_dpath%epoch)


    def save_one_sample(self, img_name='generated.png'):
        print()
        print('save one img to ', img_name)
        with torch.no_grad():
            z = torch.randn(1, 100, 1, 1, device=self.device)
            generated_img = self.model.G(z)
            save_image(generated_img.cpu(), img_name)

 