DatasetList = [u"dataset 0 - MNIST",u"dataset 1 - CIFAR 10",u"dataset 2 - CIFAR 100",u"dataset 3 - STL 10",u"dataset 4 - Fashion MNIST",u"dataset 5 - ImageNet",u"dataset 6 - EMNIST"]
Batch_sizeList = [u"4",u"8",u"16",u"32",u"64",u"128",u"256",u"512",u"1024",u"2048"]
Labeldataset1List = [u"Chiffre 0",u"Chiffre 1",u"Chiffre 2",u"Chiffre 3",u"Chiffre 4",u"Chiffre 5",u"Chiffre 6",u"Chiffre 7",u"Chiffre 8",u"Chiffre 9"]

for cnt,listedonnee in enumerate(DatasetList, 0):
    print("[%d] %s" % (cnt, listedonnee))
print("\r")    
    # CHOIX DU DATASET
optdataset = int(input("Choisissez le dataset à tester [0-%s]: " % cnt))
print("Dataset " + DatasetList[optdataset] + " sélectionné")
for cnt,listebatch in enumerate(Batch_sizeList, 0):
    print("[%d] %s" % (cnt, listebatch))
print("\r")    
    # FIN DU CHOIX DU DATASET

    # CHOIX DU BATCH SIZE
optbatch_size = int(input("Choisissez le batch size à tester [0-%s]: " % cnt))
print("Batch de " + Batch_sizeList[optbatch_size] + " sélectionné")
optbatch_size = Batch_sizeList[optbatch_size]
print("\r")  
    # FIN DU CHOIX DU BATCH SIZE
    
    # CHOIX DU NOMBRE D'EPOCH
optn_epochs = int(input("Choisissez le nombre d'EPoch : "))
print(str(optn_epochs) + " EPOCH sélectionnée")
print("\r")    
    # FIN DU CHOIX DU NOMBRE D'EPOCH
    
    # CHOIX DU LABEL à GENERER
optgenidlabel = int(input("Choisissez le label à générer [0-%s]: " % cnt))
print("Dataset " + Labeldataset1List[optgenidlabel] + " sélectionné")
print("\r")  
    # FIN DU DU LABEL à GENERER
    
    # CHOIX DU NOMBRE à GENERER
optgennumber = int(input("Entrer le nombre à générer : "))
print("Le nombre \"" + str(optgennumber) + "\" va être généré")
print("\r")      
    # FIN DU CHOIX DU NOMBRE
    
    # CHOIX DU NOMBRE D'INTERVAL à SAMPLER
optsample_interval = int(input("Entrer l'interval de génération d'image : "))
print(str(optsample_interval) + " va petre généré")
print("\r")  
    # FIN DU DU NOMBRE D'INTERVAL à SAMPLER

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
