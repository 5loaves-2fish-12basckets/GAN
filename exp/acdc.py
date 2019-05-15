import os
import sys
sys.path.append('.')

from module.DCGAN import DCGAN
from module.WACGAN import ACGAN
from module.datafunc import celeba_loader

import torch
import torchvision
from matplotlib import pyplot as plt
plt.switch_backend('agg')

torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
torch.backends.cudnn.deterministic = True 

class Maker():
    def __init__(self, dest_dir):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fixed_z = torch.randn(32, 100, 1, 1, device=self.device)
        self.fixed_z2 = torch.randn(10, 100-1, device=self.device)
        self.l = torch.LongTensor(10).fill_(0).to(self.device)
        self.l2 = torch.LongTensor(10).fill_(1).to(self.device)
        self.topil = torchvision.transforms.ToPILImage()
        self.dcmodelpath = 'dcgan.pth'
        self.acmodelpath = 'acgan.pth'
        self.dest_dir = dest_dir

    def DCgen(self):
        self.model = DCGAN() 
        self.model.load(self.dcmodelpath)
        self.model.to(self.device)
        self.make_32_graphs()

    def make_32_graphs(self):
        ## self.model is DCGAN
        with torch.no_grad():
            gen_images = self.model.G(self.fixed_z)
            gen_images = gen_images.clamp(0,1)
        fig, axes = plt.subplots(4, 8, figsize=(8, 4))
        for ax, img in zip(axes.flatten(), gen_images):
            ax.axis('off')
            pilimg = self.topil(img.cpu())
            ax.imshow(pilimg)
        plt.subplots_adjust(wspace=0, hspace=0)

        fig.text(0.5, 0.04, 'DCGAN samples', ha='center')
        filepath = os.path.join(self.dest_dir, 'fig1_2.jpg')
        plt.savefig(filepath)
        plt.close()

    def ACgen(self):
        self.model = ACGAN()
        self.model.load(self.acmodelpath)
        self.model.to(self.device)
        self.make_10_sets()

    def make_10_sets(self):
        ## self.model is ACGAN
        with torch.no_grad():
            gen_images = self.model.G(self.fixed_z2, self.l)
            gen_images2 = self.model.G(self.fixed_z2, self.l2)

        fig, axes = plt.subplots(5, 4, figsize=(4, 5))
        for i in range(5):
            for j in range(2):
                ax = axes[i][j]
                ax.axis('off')
                img = gen_images[i+j*5].cpu()
                pilimg = self.topil(img)
                ax.imshow(pilimg)

                ax = axes[i][j+2]
                ax.axis('off')
                img = gen_images2[i+j*5].cpu()
                pilimg = self.topil(img)
                ax.imshow(pilimg)

        plt.subplots_adjust(wspace=0, hspace=0)

        fig.text(0.5, 0.04, '(not smiling) ACGAN samples (smiling)', ha='center')
        filepath = os.path.join(self.dest_dir, 'fig2_2.jpg')
        plt.savefig(filepath)
        plt.close()

    def run(self):
        self.DCgen()
        self.ACgen()

if __name__ == '__main__':
    dest_dir = sys.argv[1]

    maker = Maker(dest_dir)
    maker.run()


'''
for some reason ax.imshow(numpy array(64,64,3))
will squeeze 3x3 images in the subplot in grey scale
fix by transforming tensor back to PIL image and works good

'''