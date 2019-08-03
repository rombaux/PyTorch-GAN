import argparse
import os, sys
import datetime
import time
from datetime import datetime
import numpy as np
import math
import array

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

date_string = time.strftime("%Y-%m-%d_%H-%M")
pathimagemodel = os.path.join(os.path.sep,'content','gdrive','My Drive','TFE','dataset',str(opt.dataset),date_string,'modelimage')
print ("Path of model is created as " + pathimagemodel)
os.makedirs(pathimagemodel, exist_ok=True)

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
    save_image(gen_imgs.data,  "/content/gdrive/My Drive/TFE/dataset/"+str(opt.dataset)+"/"+date_string+"/modelimage/%d.png" % batches_done, nrow=n_row, normalize=True)

fn = []

pathmodel = "/content/gdrive/My Drive/TFE/dataset/" + str(opt.dataset)
for base, dirs, files in os.walk(pathmodel):
        for file in files:
            fn.append(os.path.join(base, file))
 
fileList = [name for name in fn if name.endswith(".pth")]

for cnt, fileName in enumerate(fileList, 0):
    print("[%d] %s" % (cnt, fileName))

choice = int(input("Choisissez le modèle à tester [1-%s]: " % cnt))
print("Path of model.pth is " + fileList[choice])
pmodel = fileList[choice]

generator = Generator()
if cuda:
    generator.cuda()
generator.load_state_dict(torch.load(pmodel))

print("Génération de l'image")
sample_image(n_row=opt.n_classes, batches_done=1, date_string=date_string)
print("Image generée dans " + pathimagemodel)

'''
from IPython.display import Image, display
image = Image(filename='/content/gdrive/My Drive/TFE/dataset/1/2019-08-03_09-36/modelimage/1.png')

#image = Image.open("/content/gdrive/My Drive/TFE/dataset/1/2019-08-03_09-36/modelimage/1.png")
image.open()
'''
def upload_files():
  from google.colab import files
  uploaded = files.upload()
  for k, v in uploaded.items():
    open(k, 'wb').write(v)
  return list(uploaded.keys())

#upload_files("/content/gdrive/My Drive/TFE/dataset/1/2019-08-03_09-36/modelimage/1.png")