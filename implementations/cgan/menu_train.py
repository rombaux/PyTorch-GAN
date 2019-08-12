import time
from termcolor import colored
from IPython.display import Image

display(Image(filename='/content/PyTorch-GAN/assets/umons.jpg',width=512,height=186))

print(colored("---------------------------------------------------------------","green"))
print(colored("---    BIENVENUE SUR LA PAGE DES RESEAUX CONDITIONAL GAN    ---","green"))
print(colored("---------------------------------------------------------------","green"))
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
print(colored("---------------------------" + date_string[0:10] + "--------------------------","green"))
print(colored("-----------------------------" + date_string[11:16] + "-----------------------------","green"))
print(colored("---------------------------------------------------------------","green"))
print(colored("---                  ENTRAINEMENT DU MODELE                 ---","green"))
print(colored("---------------------------------------------------------------","green"))
display(Image(filename='/content/PyTorch-GAN/assets/reseau.jpg',width=512,height=247))
print(colored("---------------------------------------------------------------","green"))
print(colored("---                  MENU DE CONFIGURATION                  ---","green"))
print(colored("---------------------------------------------------------------","green"))
print("\r\n")

DatasetList = [u"Dataset 0 - MNIST",u"Dataset 1 - CIFAR 10",u"Dataset 2 - CIFAR 100",u"Dataset 3 - STL 10",u"Dataset 4 - Fashion MNIST",u"Dataset 5 - EMNIST"]
DatasetSize = [u"60000",u"50000",u"50000",u"5000",u"70000",u"697932"]
Batch_sizeList = [u"2",u"4",u"8",u"16",u"32",u"64",u"128",u"256",u"512",u"1024",u"2048"]
Labeldataset0List = [u"Chiffre 0",u"Chiffre 1",u"Chiffre 2",u"Chiffre 3",u"Chiffre 4",u"Chiffre 5",u"Chiffre 6",u"Chiffre 7",u"Chiffre 8",u"Chiffre 9"]
Labeldataset1List = [u"Airplane",u"Auto",u"Bird",u"Cat",u"Deer",u"Dog",u"Frog",u"Horse",u"Ship",u"Truck"]
Labeldataset2List = [u"Airplane",u"Auto",u"Bird",u"Cat",u"Deer",u"Dog",u"Frog",u"Horse",u"Ship",u"Truck"]  #TO COMPLETTTTTTTEEE
Labeldataset3List = [u"Airplane",u"Bird",u"Auto",u"Cat",u"Deer",u"Dog",u"Horse",u"Monkey",u"Ship",u"Truck"]
Labeldataset4List = [u"T-shirt/top",u"Trouser",u"Pullover",u"Dress",u"Coat",u"Sandal",u"Shirt",u"Sneaker",u"Bag",u"Ankle boot"]
Labeldataset5List = [u"Chiffre 0",u"Chiffre 1",u"Chiffre 2",u"Chiffre 3",u"Chiffre 4",u"Chiffre 5",u"Chiffre 6",u"Chiffre 7",u"Chiffre 8",u"Chiffre 9",
                    u"Lettre A",u"Chiffre B",u"Lettre C",u"Lettre D",u"Lettre E",u"Lettre F",u"Lettre G",u"Lettre H",u"Lettre I",u"Lettre J",
                    u"Lettre K",u"Lettre L",u"Lettre M",u"Lettre N",u"Lettre O",u"Lettre P",u"Lettre Q",u"Lettre R",u"Lettre S",u"Lettre T",
                    u"Lettre U",u"Lettre V",u"Lettre W",u"Lettre X",u"Lettre Y",u"Lettre Z",
                    u"Lettre a",u"Lettre b",u"Lettre c",u"Lettre d",u"Lettre e",u"Lettre f",u"Lettre g",u"Lettre h",u"Lettre i",u"Lettre j",
                    u"Lettre k",u"Lettre l",u"Lettre m",u"Lettre n",u"Lettre o",u"Lettre p",u"Lettre q",u"Lettre r",u"Lettre s",u"Lettre t",
                    u"Lettre u",u"Lettre v",u"Lettre w",u"Lettre x",u"Lettre y",u"Lettre z"]

LabeldatasetList = []

    # CHOIX DU DATASET
for cnt,listedataset in enumerate(DatasetList, 0):
    print("[%d] %s" % (cnt, listedataset))
optdataset = int(input("Choisissez le dataset à entraîner [0-%s]: " % cnt) or "0")
print(DatasetList[optdataset] + " sélectionné")
print("\r\n")
    # FIN DU CHOIX DU DATASET
 
if optdataset == 0:
    optchannel = 1
    optn_classes = 10
    optimg_size = 32

if optdataset == 1:
    optchannel = 3
    optn_classes = 10
    optimg_size = 32

if optdataset == 2:
    optchannel = 3
    optn_classes = 100
    optimg_size = 32

if optdataset == 3:
    optchannel = 3
    optn_classes = 10
    optimg_size = 96

if optdataset == 4:
    optchannel = 1
    optn_classes = 10
    optimg_size = 28

if optdataset == 5:
    optchannel = 1
    optn_classes = 62
    optimg_size = 28


# CHOIX DE LA TAILLE D'IMAGE
optimg_size = int(input("Entrer la taille de l'image (Défaut : "+ str(optimg_size) + "px x " + str(optimg_size) + "px) : ") or optimg_size)
print("L'image va être resize en " + str(optimg_size) + " pixels")
print("\r\n")      
# FIN DE LA TAILLE D'IMAGE

# CHOIX DU BATCH SIZE
for cnt,listebatch in enumerate(Batch_sizeList, 0):
    print("[%d]\t %s\t" % (cnt, listebatch) + "\t soit " + str(round(int(DatasetSize[optdataset])/int(listebatch))) + " Batches par Epoch")    
optbatch_size = int(input("Choisissez le batch size à tester [0-%s]: " % cnt))
print("Batch de " + Batch_sizeList[optbatch_size] + " sélectionné")
optbatch_size = Batch_sizeList[optbatch_size]
print("\r\n")  
# FIN DU CHOIX DU BATCH SIZE
  
# CHOIX DU NOMBRE D'INTERVAL à SAMPLER
optsample_interval = int(input("Entrer l'interval de génération d'image : "))
print("L'interval de " + str(optsample_interval) + " a été choisi")
print("\r\n")  
# FIN DU DU NOMBRE D'INTERVAL à SAMPLER

# CHOIX DU NOMBRE D'EPOCH
optn_epochs = int(input("Choisissez le nombre d'EPoch : "))
print("L'apprentissage durera " + str(optn_epochs) + " EPOCH")
print("\r\n")    
# FIN DU CHOIX DU NOMBRE D'EPOCH