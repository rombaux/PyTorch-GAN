import os, sys
fn = []
pathmodel = "/content/gdrive/My Drive/TFE/dataset/" + str(opt.dataset)
for base, dirs, files in os.walk(pathmodel):
        for file in files:
            fn.append(os.path.join(base, file))
print("Recherche dans : " + pathmodel + "\n\r") 
fileList = [name for name in fn if name.endswith("loss.png")]

for cnt, fileName in enumerate(fileList, 0):
    print("[%d] %s" % (cnt, fileName))

choice = int(input("Choisissez le compte-rendu LOSS à afficher [0-%s]: " % cnt))

from IPython.display import Image
print('Evolution des LOSS')

display(Image(filename=fileList[choice]))