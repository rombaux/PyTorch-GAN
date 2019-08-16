import os
import numpy as np
from moviepy.editor import ImageSequenceClip
from PIL import Image
#Installation instructions: 
#    pip install numpy
#    pip install moviepy
#    Moviepy needs ffmpeg tools on your system
#        (I got mine with opencv2 installed with ffmpeg support)

def create_gif(filename, array, fps=10, scale=1.0):
    """creates a gif given a stack of ndarray using moviepy
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """
    fname, _ = os.path.splitext(r'''C:\Users\Mic\tfe\modelimage\image''')   #split the extension by last period
    filename = fname + '.gif'               #ensure the .gif extension
    if array.ndim == 3:                     #If number of dimensions are 3, 
        array = array[..., np.newaxis] * np.ones(3)   #copy into the color 
                                                      #dimension if images are 
                                                      #black and white
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)
    return clip

img = Image.open(r'''C:\Users\Mic\tfe\modelimage\mot_avec_random_noize_2019-08-16_21-24_dataset5.png''').convert('RGBA')
arr = np.array(img)

imgv = Image.open(r'''C:\Users\Mic\tfe\modelimage\mot_avec_random_noize_2019-08-16_21-24_dataset5_vice.png''').convert('RGBA')
arrv = np.array(imgv)

# record the original shape
shape = arr.shape
shapev = arrv.shape
# make a 1-dimensional view of arr
flat_arr = arr.ravel()
flat_arrv = arrv.ravel()
# convert it to a matrix
vector = np.matrix(flat_arr)
vectorv = np.matrix(flat_arrv)
# do something to the vector
vector[:,::10] = 128
vectorv[:,::10] = 128
# reform a numpy array of the original shape
arr2 = np.asarray(vector).reshape(shape)
arr2v = np.asarray(vectorv).reshape(shapev)
# make a PIL image
img2 = Image.fromarray(arr2, 'RGBA')
img2.show()

img2v = Image.fromarray(arr2v, 'RGBA')
img2v.show()