import os, sys
fn = []
pathmodel = "/content/gdrive/My Drive/TFE/dataset/"
for base, dirs, files in os.walk(pathmodel):
        for file in files:
            fn.append(os.path.join(base, file))
print("Recherche dans : " + pathmodel + "\n\r") 
fileList = [(0, 'dataset0'),(1,'dataset1')]

for cnt, fileName in enumerate(fileList, 0):
    print("[%d] %s" % (cnt, fileName))

choice = int(input("Choisissez le dataset à tester [0-%s]: " % cnt))

print(choice)


