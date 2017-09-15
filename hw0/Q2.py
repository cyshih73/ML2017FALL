import sys
from PIL import Image
img = Image.open(sys.argv[1])
width, height = img.size
for y in range(height):
    for x in range(width):
        rgb = img.getpixel((x,y))
        rgb = (int(rgb[0]/2), int(rgb[1]/2), int(rgb[2]/2));
        #R G B
        img.putpixel((x,y), rgb)
 
img.save( "Q2.png" )