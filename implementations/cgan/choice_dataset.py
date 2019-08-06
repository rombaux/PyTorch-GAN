DatasetList = [u"dataset 0 - MNIST",u"dataset 1 - CIFAR 10",u"dataset 2 - CIFAR 100",u"dataset 3 - STL 10",u"dataset 4 - Fashion MNIST",u"dataset 5 - ImageNet",u"dataset 6 - EMNIST"]

for cnt,listedonnee in enumerate(DatasetList, 0):
    print("[%d] %s" % (cnt, listedonnee))
print("\r")    

optdataset = int(input("Choisissez le dataset à tester [0-%s]: " % cnt))

print("Dataset " + DatasetList[optdataset] + " sélectionné")

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
