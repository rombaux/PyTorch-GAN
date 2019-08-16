from images2gif import writeGif
from images2gif import readGif as readGif
from PIL import Image
import os

path = r'''C:\Users\Mic\tfe\modelimage'''

file_names = sorted((fn for fn in os.listdir(path) if fn.endswith('.png')))
#['animationframa.png', 'animationframb.png', ...] "

images = [Image.open(fn) for fn in file_names]

size = (150,150)
for im in images:
    im.thumbnail(size, Image.ANTIALIAS)

filename = r'''C:\Users\Mic\tfe\modelimage\my_gif.GIF'''
writeGif(filename, images, duration=0.2)