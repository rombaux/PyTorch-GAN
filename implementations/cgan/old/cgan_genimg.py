import argparse
import os, sys
import datetime
import time
from datetime import datetime
import numpy as np
import math
import csv
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import torch.optim as optim

os.makedirs("images", exist_ok=True)

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
parser.add_argument("--dataset", type=int, default=0, help="choice of dataset - Nmist = 0 - cifar10 = 1 - cifar100 = 2 - stl10 = 3")
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

pathimage = os.path.join(os.path.sep,'content','gdrive','My Drive','TFE','dataset',str(opt.dataset),date_string,'gen09')
print ("Path \"generateur complet\" is created as " + pathimage)
os.makedirs(pathimage, exist_ok=True)

pathimage = os.path.join(os.path.sep,'content','gdrive','My Drive','TFE','dataset',str(opt.dataset),date_string,str(opt.genidlabel))
print ("Path \"generateur label choisi\" is created as " + pathimage)
os.makedirs(pathimage, exist_ok=True)

pathimage = os.path.join(os.path.sep,'content','gdrive','My Drive','TFE','dataset',str(opt.dataset),date_string,str(opt.gennumber))
print ("Path \"generateur du nombre\"is created as " + pathimage)
os.makedirs(pathimage, exist_ok=True)

pathmodel = os.path.join(os.path.sep,'content','gdrive','My Drive','TFE','dataset',str(opt.dataset),date_string,'model')
print ("Path of model is created as " + pathmodel)
os.makedirs(pathmodel, exist_ok=True)

pathmodel = os.path.join(os.path.sep,'content','gdrive','My Drive','TFE','dataset',str(opt.dataset),date_string,'loss')
print ("Path of model is created as " + pathmodel)
os.makedirs(pathmodel, exist_ok=True)

print("Dataset n: " + str(opt.dataset) + " selected and " + str(opt.n_classes) + " classes used")

fichier = open("/content/gdrive/My Drive/TFE/dataset/"+str(opt.dataset)+"/"+date_string+"/" + "config.txt", "a")
fichier.write(str(opt))
fichier.close()
print("Fichier de configuration créé")

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
     
        PATCH = "/content/gdrive/My Drive/TFE/dataset/"+str(opt.dataset)+"/"+date_string+"/model/modelg.pth"
        torch.save(self.model.state_dict(prefix='model'), PATCH)  

        # Print model's state_dict
        print( "Model's state_dict:" )
        for param_tensor in self.model.state_dict ():
            print(param_tensor , " \t " , self.model.state_dict ()[ param_tensor ] . size ())

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity

# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

if opt.dataset == 0:                # MNIST
    # Configure data loader
    os.makedirs("../../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )

if opt.dataset == 1:                # CIFAR 10
    # Configure data loader
    os.makedirs("../../data/cifar10", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "../../data/cifar10",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize(( 0.5 , 0.5 , 0.5 ), ( 0.5 , 0.5 , 0.5 ))]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )

if opt.dataset == 2:                # CIFAR 100
    # Configure data loader
    os.makedirs("../../data/cifar100", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.CIFAR100(
            "../../data/cifar100",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize(( 0.5 , 0.5 , 0.5 ), ( 0.5 , 0.5 , 0.5 ))]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )

if opt.dataset == 3:                # STL 10
    # Configure data loader
    os.makedirs("../../data/STL10", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.STL10(
            "../../data/STL10",
            split='train',
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(),transforms.Normalize(( 0.5 , 0.5 , 0.5 ), ( 0.5 , 0.5 , 0.5 ))]
            ),
            #target_transform=None,
            download=True
            ),
        
        batch_size=opt.batch_size,
        shuffle=True,
    )  

if opt.dataset == 4:                # FASHION MNIST
    # Configure data loader
    os.makedirs("../../data/FashionMNIST", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "../../data/FashionMNIST",
            train=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))]
            ),
            #target_transform=None,
            download=True
            ),
        
        batch_size=opt.batch_size,
        shuffle=True,
    )   

if opt.dataset == 5:
    print("image redimensionné à " + str(opt.img_size))
    # Configure data loader
    os.makedirs("../../data/church_train", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.church_train(
            "../../data/church_train",
            classes='church_train',
            transform=transforms.Compose(
                [transforms.Resize((opt.img_size,opt.img_size)), transforms.ToTensor(), transforms.Normalize(( 0.5 , 0.5 , 0.5 ), ( 0.5 , 0.5 , 0.5 ))]
            ),
            #target_transform=True,
            download=True
            ),
        
        batch_size=opt.batch_size,
        shuffle=True,
    )    

if opt.dataset == 6:            # DATASET EMNIST
    # Configure data loader
    os.makedirs("../../data/EMNIST", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.EMNIST(
            "../../data/EMNIST",
            split='byclass',
            transform=transforms.Compose([
                transforms.Resize(opt.img_size),
                lambda img: transforms.functional.rotate(img, -90),
                lambda img: transforms.functional.hflip(img),
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))
                ]),
            #target_transform=True,
            download=True
            ),
        
        batch_size=opt.batch_size,
        shuffle=True,
    )            

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

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
    save_image(gen_imgs.data,  "/content/gdrive/My Drive/TFE/dataset/"+str(opt.dataset)+"/"+date_string+"/gen09/full_interval_%s.png" % (str(batches_done).zfill(4)), nrow=n_row, normalize=True)
    
def sample_label_id_image(n_row, batches_done,date_string):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    if opt.genidlabel < 10:
        save_image(gen_imgs.data[opt.genidlabel],  "/content/gdrive/My Drive/TFE/dataset/"+str(opt.dataset)+"/"+date_string+"/"+str(opt.genidlabel)+"/gen_"+str(opt.genidlabel)+"_interval_%s.png" % (str(batches_done).zfill(4)), nrow=n_row, normalize=True)
    if opt.gennumber > 0:
        toto = opt.gennumber
        numbre =[]
        for a in str(toto):
            numbre.append(gen_imgs.data[int(a)])
        pathimagegen = "/content/gdrive/My Drive/TFE/dataset/"+str(opt.dataset)+"/"+date_string+"/"+str(opt.gennumber)+"/gen"+str(opt.gennumber)+"_interval_%s.png" % (str(batches_done).zfill(4))
        print("image save in" + pathimagegen)
        save_image(numbre, pathimagegen , nrow=n_row, normalize=True)
        print("nombre : "+str(opt.gennumber)+" generated")


# ----------
#  Training
# ----------
cpt = 0

dloss = []
gloss = []
drealloss = []
dfakeloss = []

xdloss = []
xgloss = []

#dloss.append('DLoss')
#gloss.append('GLoss')

#xdloss.append('XDLoss')
#xgloss.append('XGLoss')

compteur = 0
sample_image(n_row=opt.n_classes, batches_done=compteur, date_string=date_string)
sample_label_id_image(n_row=opt.n_classes, batches_done=compteur, date_string=date_string)
print("image sauvée" + str(compteur) + ".png")

for epoch in range(opt.n_epochs):

    print("image redimensionné à " + str(opt.img_size))

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
            heure = time.strftime("%Y-%m-%d_%H-%M")
            a = heure[11:13]
            a= str(a)
            b = str(int(a) + 2)
            b = b.zfill(2)
            list1 = list(heure)
            list1[11] = b[0]
            list1[12] = b[1]
            attime = ''.join(list1)
            
            print(attime + " [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))
            a = float(d_loss.item())
            b = float(g_loss.item())
            
            dloss.append(a)
            gloss.append(b)

            xdloss.append(batches_done)
            xgloss.append(batches_done)

            # Procedure des création des log des Loss - Fichier.csv + Graphe
            #  
            with open('/content/gdrive/My Drive/TFE/dataset/' + str(opt.dataset) + '/' + date_string +'/loss/loss.csv', mode='w') as loss_file:
                loss_writer = csv.writer(loss_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                loss_writer.writerows(zip(*[dloss, gloss]))

            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(3, 1, 1)

            ax.plot(dloss, color='xkcd:dark pink')
            ax.plot(gloss, color='xkcd:navy blue')

            ax.set_xlabel("Samples")
            ax.set_ylabel("LOSS")
            ax.set_title("Evolution des Loss" + str(opt.sample_interval))
            plt.legend(['Discriminator Loss', 'Generator Loss'])

            plt.savefig("/content/gdrive/My Drive/TFE/dataset/" + str(opt.dataset) + '/' + date_string + "/loss/loss.png")

            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(3, 1, 1)

            ax.plot(xdloss,dloss, color='xkcd:dark pink')
            ax.plot(xgloss,gloss, color='xkcd:navy blue')

            ax.set_xlabel("Samples")
            ax.set_ylabel("LOSS")
            ax.set_title("Evolution des Loss avec un interval de " + str(opt.sample_interval) + " images")
            plt.legend(['Discriminator Loss', 'Generator Loss'])

            plt.savefig("/content/gdrive/My Drive/TFE/dataset/" + str(opt.dataset) + '/' + date_string + "/loss/loss_dloss_xdloss.png")

            plt.close('all')

# --------->indentation pour  sample par batch 
            compteur = compteur + 1
            sample_image(n_row=opt.n_classes, batches_done=compteur, date_string=date_string)
            sample_label_id_image(n_row=opt.n_classes, batches_done=compteur, date_string=date_string)
            print("Les images pour l'interval n° : " + str(compteur).zfill(4) + " et batches_done = " + str(batches_done).zfill(4))
            print("sauvée dans /content/gdrive/My Drive/TFE/dataset/"+str(opt.dataset)+"/"+date_string + "/")
 
    PATCH = "/content/gdrive/My Drive/TFE/dataset/"+str(opt.dataset)+"/"+date_string+"/model/"+"model_from_epoch_" + str(epoch).zfill(4) + ".pth"
    torch.save(generator.state_dict(), PATCH)
    print("Model saved in "+str(PATCH))       