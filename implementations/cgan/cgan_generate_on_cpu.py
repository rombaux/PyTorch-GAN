import argparse
import os, sys
import os.path
import datetime
import time
from datetime import datetime
import numpy as np
import math
import array
import cv2

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

cuda = True if torch.cuda.is_available() else False
print("torch cuda is available => " + str(torch.cuda.is_available()))

# Correction l'heure en GMT+2  - A corriger pour 24h et 25h mais fonctionne
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

def Vice(pic, name):
    pic = (255 - pic)
    cv2.imwrite(name, pic)

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
    print("z == " + str(z))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
#    print("labels == " + str(labels))
    labels = Variable(LongTensor(labels))
 #   print("labels == " + str(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data,  r'''C:\Users\Mic\tfe\modelimage\full_avec_random_noize_'''+ str(date_string) + '''_dataset''' + str(opt.dataset) + '''.png''', nrow=n_row, normalize=True)

def sample_label_id_image_without_random_noise(n_row, batches_done,date_string):
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
    save_image(word, r'''C:\Users\Mic\tfe\modelimage\mot_sans_random_noize_''' + str(date_string) + '''_dataset''' + str(opt.dataset) + '''.png''', nrow=len(str(toto)), normalize=True)
    print("Suite : "+(opt.genword)+" générée")

def sample_label_id_image_with_random_noise(n_row, batches_done,date_string):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise

    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))

    toto = opt.genword 
    word =[]
    for a in str(toto):
        z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
        gen_imgs = generator(z, labels)
        a = dictotodataset5.get(a)
        word.append(gen_imgs.data[int(a)])
    save_image(word, r'''C:\Users\Mic\tfe\modelimage\mot_avec_random_noize_''' + str(date_string) + '''_dataset''' + str(opt.dataset) + '''.png''', nrow=len(str(toto)), normalize=True)
    print("Suite : "+(opt.genword)+" générée")

# Recherche du modèle
fn = []
cnt = 0

pathmodel = r'''C:\Users\Mic\tfe'''
for base, dirs, files in os.walk(pathmodel):
        for file in files:
            fn.append(os.path.join(base, file))
print("Recherche dans : " + pathmodel + "\n\r") 
fileList = [name for name in fn if name.endswith(".pth")]

for cnt, fileName in enumerate(fileList, 0):
    print("[%d] %s" % (cnt, fileName))

choice = int(input("Choisissez le modèle à tester [0-%s]: " % cnt))

print("Path of Model Choise is " + fileList[choice])
pmodel = fileList[choice]

a = os.path.dirname(pmodel)
b = os.path.dirname(a)

#pathconfig = r'''C:\Users\Mic\tfe\config.txt'''
#fichier = open(pathconfig, "r")
#file_config = fichier.read()
#print("Config : " + file_config)
#index_of_img_size = file_config.find('img_size')
#index_of_latent_dim = file_config.find('latent_dim')
#print("Attention, la taille de l'image dans le Training est de " + file_config[(index_of_img_size+9):(index_of_latent_dim-2)] + " pixels")         
#taille_img_train = int(file_config[(index_of_img_size+9):(index_of_latent_dim-2)])
#fichier.close()

img_shape = (opt.channels, opt.img_size, opt.img_size)
print(img_shape)

print("dataset : " + str(opt.dataset))
print("Size of image : " + str(opt.img_size))

generator = Generator()
if cuda:
    generator.cuda()
generator.load_state_dict(torch.load(pmodel, map_location='cpu'))

print("Génération de l'image")

pathimagemodel = r'''C:\Users\Mic\tfe'''
print ("Création du Path : " + pathimagemodel)
os.makedirs(pathimagemodel, exist_ok=True)

sample_image(n_row=opt.n_classes, batches_done = 1, date_string=date_string)
sample_label_id_image_with_random_noise(n_row=opt.n_classes, batches_done = 1, date_string=date_string)
sample_label_id_image_without_random_noise(n_row=opt.n_classes, batches_done = 1, date_string=date_string)
print("Image generée dans " + pathimagemodel + '\modelimage')
         
image = r'''C:\Users\Mic\tfe\modelimage\mot_avec_random_noize_''' + str(date_string) + '''_dataset''' + str(opt.dataset) + '''.png'''
img = cv2.imread(image)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
Vice(gray, r'''C:\Users\Mic\tfe\modelimage\mot_avec_random_noize_''' + str(date_string) + '''_dataset''' + str(opt.dataset) + '''_vice.png''')

print("Ok")

# Pour tester dataset 0
#cgan_generate_on_cpu.py --dataset 0 --channels 1 --img_size 32 --n_classes 10 --genword 123456789

# Pour tester dataset 1
#cgan_generate_on_cpu.py --dataset 1 --channels 3 --img_size 32 --n_classes 10 --genword 123456789

# Pour tester dataset 2
#cgan_generate_on_cpu.py --dataset 2 --channels 3 --img_size 32 --n_classes 100 --genword 123456789

# Pour tester dataset 3
#cgan_generate_on_cpu.py --dataset 3 --channels 3 --img_size 32 --n_classes 10 --genword 123456789

# Pour tester dataset 4
#cgan_generate_on_cpu.py --dataset 4 --channels 1 --img_size 28 --n_classes 10 --genword 123456789

# Pour tester dataset 5
#cgan_generate_on_cpu.py --dataset 5 --channels 1 --img_size 28 --n_classes 62 --genword MERCIPourVotreAttention