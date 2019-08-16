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

print("gen le opt.dataset choisi est " + str(opt.dataset))
print("gen le opt.word choisi est " + opt.genword) 
print("gen le dataset choisi est " + str(optdataset))
print("gen le mot choisi est " + optgenword) 

parser = argparse.ArgumentParser()
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--genword", type=str, default=0, help="generate number")
parser.add_argument("--dataset", type=int, default=0, help="choice of dataset - Nmist = 0 - cifar10 = 1 - cifar100 = 2")
opt = parser.parse_args()
print(opt)



dictotodataset0 = { '0':'0', '1':'1', '2':'2', '3':'3', '4':'4', '5':'5', '6':'6', '7':'7', '8':'8', '9':'9'}
dictotodataset5 = { '0':'0', '1':'1', '2':'2', '3':'3', '4':'4', '5':'5', '6':'6', '7':'7', '8':'8', '9':'9',
                    'A':'10','B':'11','C':'12','D':'13','E':'14','F':'15','G':'16','H':'17','I':'18','J':'19',
                    'K':'20','L':'21','M':'22','N':'23','O':'24','P':'25','Q':'26','R':'27','S':'28','T':'29',
                    'U':'30','V':'31','W':'32','X':'33','Y':'34','Z':'35','a':'36','b':'37','c':'38','d':'39',
                    'e':'40','f':'41','g':'42','h':'43','i':'44','j':'45','k':'46','l':'47','m':'48','n':'49',
                    'o':'50','p':'51','q':'52','r':'53','s':'54','t':'55','u':'56','v':'57','w':'58','x':'59',
                    'y':'60','z':'61'}

#img_shape = (opt.channels, opt.img_size, opt.img_size)
#print(img_shape)
opt.genword = optgenword
opt.dataset = optdataset

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

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def sample_image(n_row, batches_done,date_string):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    #print("z == " + str(z))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    #print("labels == " + str(labels))
    labels = Variable(LongTensor(labels))
    #print("labels == " + str(labels))
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
    toto = opt.genword
    word =[]
    for a in str(toto):
        a = dictotodataset5.get(a)
        word.append(gen_imgs.data[int(a)])
    save_image(word, b + "/modelimage/gen_number_" + opt.genword + "_" + date_string + "_%s.png" % (str(batches_done).zfill(4)), nrow=n_row, normalize=True)
    src = b +'/modelimage/gen_number_' + opt.genword + '_' + date_string + '_0001.png'
    dst = '/content/gdrive/My Drive/TFE/dataset/modelimage/gen_multiple_0001.png'
    copyfile(src,dst)        
    print("Suite : "+ opt.genword +" générée")


# Recherche du modèle
fn = []
cnt = 0
fndir = []

pathmodel = "/content/gdrive/My Drive/TFE/dataset/" + str(opt.dataset)
for base, dirs, files in os.walk(pathmodel):
        for file in files:
            #print("base :" + base)
            #print("dirs" + str(dirs))
            #fn.append(os.path.join(base, file))
            pathallmodel = os.path.dirname(os.path.join(base, file))
            if pathallmodel not in fndir:
                fndir.append(os.path.dirname(os.path.join(base, file)))
print("Recherche dans : " + pathmodel + "\n\r") 
#fileList = [name for name in fn if name.endswith(".pth")]
DirList = [name for name in fndir if name.endswith("model")]

for idx,DirName in enumerate(DirList):
    for base, dirs, files in os.walk(str(DirName)):
        fntemp = []
        for file in files:
            #print("base :" + base)
            #print("dirs" + str(dirs))
            fntemp.append(os.path.join(base, file))
        fntemp.reverse() 
        for i in range(0,min(9,len(fntemp))):
            fn.append(fntemp[i])


            
fileList = [name for name in fn if name.endswith(".pth")]

for cnt, fileName in enumerate(fileList, 0):
    print("[%d] %s" % (cnt, fileName))

choice = int(input("Choisissez le modèle à tester [0-%s]: " % cnt))

print("Path of Model Choise is " + fileList[choice])
pmodel = fileList[choice]

a = os.path.dirname(pmodel)
b = os.path.dirname(a)
#print("Root path of model.pth is " + str(a))
#print("Root path of directory model is " + str(b))

pathconfig = b + "/config.txt"
fichier = open(pathconfig, "r")
file_config = fichier.read()
print("Config : " + file_config)
index_of_img_size = file_config.find('img_size')
#if index_of_img_size == -1:
#    print('Not Found')
#else:
#    print("Found at index" + str(index_of_img_size))
index_of_latent_dim = file_config.find('latent_dim')
#if index_of_latent_dim == -1:
#    print('Not Found')
#else:
#    print("Found at index" + str(index_of_latent_dim))
print("Attention, la taille de l'image dans le Training est de " + file_config[(index_of_img_size+9):(index_of_latent_dim-2)] + " pixels")         
taille_img_train = int(file_config[(index_of_img_size+9):(index_of_latent_dim-2)])
fichier.close()

img_shape = (opt.channels, taille_img_train, taille_img_train)
print("Dimension de l'image (channel,size x, size y) : " + str(img_shape))

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

