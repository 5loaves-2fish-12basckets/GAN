# -*- coding: utf-8 -*-
"""trains ACGAN/DCGAN for CELEBA    
__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""
import os
import sys
sys.path.append('.')
import argparse

from module.trainer import Trainer
from module.ACTrainer import ACTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasktype', type=str, default='dc', help='ac or dc')
    parser.add_argument('--taskname', type=str, default='DCGAN', help='taskname for model saving etc')
    parser.add_argument('--resume', type=bool, default=None, help='resume or not')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--img_channel_num', type=int, default=3)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--img_size', type=int, default=64)
    args = parser.parse_args()
    print('tasktype', args.tasktype, 'taskname', args.taskname, '# epochs', args.epochs, 'batch_size', args.batch_size)
    
    if not os.path.exists('ckpt/%s'%args.taskname):
        os.mkdir('ckpt/%s'%args.taskname)
        print('created', 'ckpt/%s'%args.taskname)

    if args.tasktype == 'dc':
        trainer = Trainer(args)
    else:
        trainer = ACTrainer(args)

    trainer.train()

if __name__ == '__main__':
    main()
