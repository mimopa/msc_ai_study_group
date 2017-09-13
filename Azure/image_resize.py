import os
import glob
from PIL import Image

files = glob.glob('./data/*.jpg')

for f in files:
    img = Image.open(f)
    img_resize = img.resize((int(img.width/2), int(img.height/2)))
    ftitle, fext = os.path.splitext(f)
    img_resize.save(ftitle + '_half' + fext)