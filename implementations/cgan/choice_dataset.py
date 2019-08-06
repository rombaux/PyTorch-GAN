print("\r\n")
print("-------------------------------------------------")
print("BIENVENUE SUR LA PAGE DES RESEAUX CONDITIONAL GAN")    
print("-------------------------------------------------")   
print("\r\n")
print("-------------------------------------------------")
print("---           MENU DE CONFIGURATION           ---")    
print("-------------------------------------------------")   
print("\r\n")

DatasetList = [u"dataset 0 - MNIST",u"dataset 1 - CIFAR 10",u"dataset 2 - CIFAR 100",u"dataset 3 - STL 10",u"dataset 4 - Fashion MNIST",u"dataset 5 - ImageNet",u"dataset 6 - EMNIST"]
Batch_sizeList = [u"4",u"8",u"16",u"32",u"64",u"128",u"256",u"512",u"1024",u"2048"]
Labeldataset0List = [u"Chiffre 0",u"Chiffre 1",u"Chiffre 2",u"Chiffre 3",u"Chiffre 4",u"Chiffre 5",u"Chiffre 6",u"Chiffre 7",u"Chiffre 8",u"Chiffre 9"]
Labeldataset1List = [u"Airplane",u"Auto",u"Bird",u"Cat",u"Deer",u"Dog",u"Frog",u"Horse",u"Ship",u"Truck"]

Labeldataset3List = [u"Airplane",u"Bird",u"Auto",u"Cat",u"Deer",u"Dog",u"Horse",u"Monkey",u"Ship",u"Truck"]
Labeldataset4List = [u"T-shirt/top",u"Trouser",u"Pullover",u"Dress",u"Coat",u"Sandal",u"Shirt",u"Sneaker",u"Bag",u"Ankle boot"]

Labeldataset6List = [u"Chiffre 0",u"Chiffre 1",u"Chiffre 2",u"Chiffre 3",u"Chiffre 4",u"Chiffre 5",u"Chiffre 6",u"Chiffre 7",u"Chiffre 8",u"Chiffre 9",
                    u"Chiffre A",u"Chiffre B",u"Chiffre C",u"Chiffre D",u"Chiffre E",u"Chiffre F",u"Chiffre G",u"Chiffre H",u"Chiffre I",u"Chiffre J",
                    u"Chiffre K",u"Chiffre L",u"Chiffre M",u"Chiffre N",u"Chiffre O",u"Chiffre P",u"Chiffre Q",u"Chiffre R",u"Chiffre S",u"Chiffre T",
                    u"Chiffre U",u"Chiffre V",u"Chiffre W",u"Chiffre X",u"Chiffre Y",u"Chiffre Z",
                    u"Chiffre a",u"Chiffre b",u"Chiffre c",u"Chiffre d",u"Chiffre e",u"Chiffre f",u"Chiffre g",u"Chiffre h",u"Chiffre i",u"Chiffre j",
                    u"Chiffre k",u"Chiffre l",u"Chiffre m",u"Chiffre n",u"Chiffre o",u"Chiffre p",u"Chiffre q",u"Chiffre r",u"Chiffre s",u"Chiffre t",
                    u"Chiffre u",u"Chiffre v",u"Chiffre w",u"Chiffre x",u"Chiffre y",u"Chiffre z",
                    ]
                    
LabeldatasetList = []

    # CHOIX DU DATASET
for cnt,listedataset in enumerate(DatasetList, 0):
    print("[%d] %s" % (cnt, listedataset))
optdataset = int(input("Choisissez le dataset à tester [0-%s]: " % cnt))
print("Dataset " + DatasetList[optdataset] + " sélectionné")
print("\r\n")
    # FIN DU CHOIX DU DATASET

    # CHOIX DU BATCH SIZE
for cnt,listebatch in enumerate(Batch_sizeList, 0):
    print("[%d] %s" % (cnt, listebatch))     
optbatch_size = int(input("Choisissez le batch size à tester [0-%s]: " % cnt))
print("Batch de " + Batch_sizeList[optbatch_size] + " sélectionné")
optbatch_size = Batch_sizeList[optbatch_size]
print("\r\n")  
    # FIN DU CHOIX DU BATCH SIZE
    
    # CHOIX DU LABEL à GENERER
if optdataset == 0 : LabeldatasetList = Labeldataset0List
if optdataset == 1 : LabeldatasetList = Labeldataset1List
if optdataset == 2 : LabeldatasetList = Labeldataset2List
if optdataset == 3 : LabeldatasetList = Labeldataset3List
if optdataset == 4 : LabeldatasetList = Labeldataset4List
if optdataset == 5 : LabeldatasetList = Labeldataset5List
if optdataset == 6 : LabeldatasetList = Labeldataset6List

for cnt,listelabel in enumerate(LabeldatasetList, 0):
    print("[%d] %s" % (cnt, listelabel))    
optgenidlabel = int(input("Choisissez le label à générer [0-%s]: " % cnt))
print("Le label \"" + LabeldatasetList[optgenidlabel] + "\" a été sélectionné")
print("\r\n")  
    # FIN DU DU LABEL à GENERER
    
    # CHOIX DU NOMBRE à GENERER
optgennumber = int(input("Entrer le nombre à générer : "))
print("Le nombre \"" + str(optgennumber) + "\" va être généré")
print("\r\n")      
    # FIN DU CHOIX DU NOMBRE
    
    # CHOIX DU NOMBRE D'INTERVAL à SAMPLER
optsample_interval = int(input("Entrer l'interval de génération d'image : "))
print(str(optsample_interval) + " va petre généré")
print("\r\n")  
    # FIN DU DU NOMBRE D'INTERVAL à SAMPLER
    
    # CHOIX DU NOMBRE D'EPOCH
optn_epochs = int(input("Choisissez le nombre d'EPoch : "))
print(str(optn_epochs) + " EPOCH sélectionnée")
print("\r\n")    
    # FIN DU CHOIX DU NOMBRE D'EPOCH    

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
    optchannel = 3
    optn_classes = 1000
    optimg_size = 256

if optdataset == 6:
    optchannel = 1
    optn_classes = 62
    optimg_size = 28    
