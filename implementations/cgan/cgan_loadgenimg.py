import argparse
import os, sys
import datetime
import time
from datetime import datetime
import numpy as np
import math

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
parser.add_argument("--n_epochs", type=int, default=2, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
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
#        PATCH = "/content/gdrive/My Drive/TFE/model/modelg.pth"
 #       torch.save(self.model.state_dict(), PATCH)  

        # Print model's state_dict
        
#        print( "Model's state_dict of generator :" )
#        for param_tensor in self.model.state_dict ():
#            print(param_tensor , " \t " , self.model.state_dict ()[ param_tensor ] . size ())
        
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
    z = Variable(FloatTensor(np.random.normal(0.5, 0.5, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    #labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(pmodel))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data,  "/content/gdrive/My Drive/TFE/dataset/"+str(opt.dataset)+"/"+date_string+"/modelimage/%d.png" % batches_done, nrow=n_row, normalize=True)


pmodel = "/content/gdrive/My Drive/TFE/dataset/0/2019-08-02_01-27/model/model_738205.pth"
print ("Path is " + pmodel)
model_weights = torch.load(pmodel)
print(type(model_weights))
for k in model_weights:
    print(k)

print("ICIIIIIIIIII")
'''
device = torch.device("cuda")
model = Generator()
model.load_state_dict(torch.load(pmodel))
print("Load Model in " + pmodel)
'''
model_weights.to(torch.device('cuda'))
model_weights.to(device)


print("Generation image")
sample_image(n_row=opt.n_classes, batches_done=1, date_string=date_string)
print("Image generee")

# torch.save(generator.model.state_dict(), PATCH)

# ----------
#  Training
# ----------

'''
for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))
            #sample_image(n_row=10, batches_done=batches_done, date_string=date_string)
            #sample_label_id_image(n_row=10, batches_done=batches_done, date_string=date_string)
            sample_image(n_row=opt.n_classes, batches_done=batches_done, date_string=date_string)
            sample_label_id_image(n_row=opt.n_classes, batches_done=batches_done, date_string=date_string)

    PATCH = "/content/gdrive/My Drive/TFE/model/model_dataset" + str(opt.dataset) + "_" +str(batches_done)+".pth"
    torch.save(generator.model.state_dict(), PATCH)
    print("Model saved in "+str(PATCH))           
                    
'''
