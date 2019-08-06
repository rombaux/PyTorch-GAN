
fileList = [(0, 'dataset0'),(1,'dataset1')]

for cnt in enumerate(fileList, 0):
    print("[%d] %s" % (cnt, fileName))

choice = int(input("Choisissez le dataset à tester [0-%s]: " % cnt))

print(choice)


