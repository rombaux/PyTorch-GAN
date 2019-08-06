DatasetList = [u"dataset 0 - MNIST",u"dataset 1 - CIFAR 10",u"dataset 2 - CIFAR 100",u"dataset 3 - STL 10",u"dataset 4 - Fashion MNIST",u"dataset 5 - ImageNet",u"dataset 6 - EMNIST"]

for cnt,listedonnee in enumerate(DatasetList, 0):
    print("[%d] %s" % (cnt, listedonnee))
print("\r")    

choice = int(input("Choisissez le dataset à tester [0-%s]: " % cnt))

print("Dataset " + DatasetList[choice] + " sélectionné")