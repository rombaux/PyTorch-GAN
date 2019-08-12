from IPython.display import Image
import time
time.sleep(2)
print('\r\nGrille de tous les labels du dataset')
display(Image(filename='/content/gdrive/My Drive/TFE/dataset/modelimage/full_0001.png'))
print('\r\nMot Choisi : ')
print("Suite : "+str(opt.gennumber)+" générée")
display(Image(filename='/content/gdrive/My Drive/TFE/dataset/modelimage/gen_multiple_0001.png'))