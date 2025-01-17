import argparse
import os, sys
import os.path
import datetime
import time
from datetime import datetime
import numpy as np
import math
import array

from shutil import copyfile

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--genidlabel", type=int, default=10, help="generate image whit label")
parser.add_argument("--gennumber", type=int, default=0, help="generate number")
parser.add_argument("--dataset", type=int, default=0, help="choice of dataset - Nmist = 0 - cifar10 = 1 - cifar100 = 2")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
print("torch cuda is available => " + str(torch.cuda.is_available()))

heure = time.strftime("%Y-%m-%d_%H-%M")
a = heure[11:13]
a= str(a)
b = str(int(a) + 2)
b = b.zfill(2)
date_string = heure
list1 = list(date_string)
list1[11] = b[0]
list1[12] = b[1]
date_string = ''.join(list1)

print("Dataset n: " + str(opt.dataset) + " selected and " + str(opt.n_classes) + " classes used")

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
      
    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img

# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()

if cuda:
    generator.cuda()
    adversarial_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def sample_image(n_row, batches_done,date_string):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, b + "/modelimage/full_" + date_string + "_%s.png" % (str(batches_done).zfill(4)), nrow=n_row, normalize=True)
    src = b + '/modelimage/full_' + date_string + '_0001.png'
    dst = '/content/gdrive/My Drive/TFE/dataset/modelimage/full_0001.png'
    copyfile(src,dst)

def sample_label_id_image(n_row, batches_done,date_string):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    if opt.genidlabel < 10:
        save_image(gen_imgs.data[opt.genidlabel], b + "/modelimage/gen_label_" + str(opt.genidlabel) + "_" + date_string + "_%s.png" % (str(batches_done).zfill(4)), nrow=n_row, normalize=True)
        src = b + '/modelimage/gen_label_' + str(opt.genidlabel) + '_' + date_string + '_0001.png'
        dst = '/content/gdrive/My Drive/TFE/dataset/modelimage/gen_label_id_0001.png'
        copyfile(src,dst)
        print("label : "+str(opt.genidlabel)+" generated")
    if opt.gennumber > 0:
        toto = opt.gennumber
        numbre =[]
        for a in str(toto):
            numbre.append(gen_imgs.data[int(a)])
        save_image(numbre, b + "/modelimage/gen_number_" + str(opt.gennumber) + "_" + date_string + "_%s.png" % (str(batches_done).zfill(4)), nrow=n_row, normalize=True)
        src = b +'/modelimage/gen_number_' + str(opt.gennumber) + '_' + date_string + '_0001.png'
        dst = '/content/gdrive/My Drive/TFE/dataset/modelimage/gen_multiple_0001.png'
        copyfile(src,dst)        
        print("nombre : "+str(opt.gennumber)+" generated")


# Recherche du modèle
fn = []
cnt = 0

pathmodel = "/content/gdrive/My Drive/TFE/dataset/" + str(opt.dataset)
for base, dirs, files in os.walk(pathmodel):
        for file in files:
            fn.append(os.path.join(base, file))
print("Recherche dans : " + pathmodel + "\n\r") 
fileList = [name for name in fn if name.endswith(".pth")]

for cnt, fileName in enumerate(fileList, 0):
    print("[%d] %s" % (cnt, fileName))

choice = int(input("Choisissez le modèle à tester [0-%s]: " % cnt))

print("Path of model.pth is " + fileList[choice])
pmodel = fileList[choice]

a = os.path.dirname(pmodel)
b = os.path.dirname(a)

print("Root path of model.pth is " + str(a))
print("Root path of model is " + str(b))

generator = Generator()
if cuda:
    generator.cuda()
generator.load_state_dict(torch.load(pmodel))

print("Génération de l'image")


pathimagemodel = b + "/modelimage"
print ("Création du Path : " + pathimagemodel)
os.makedirs(pathimagemodel, exist_ok=True)

sample_image(n_row=opt.n_classes, batches_done = 1, date_string=date_string)
sample_label_id_image(n_row=opt.n_classes, batches_done = 1, date_string=date_string)
print("Image generée dans " + pathimagemodel)

