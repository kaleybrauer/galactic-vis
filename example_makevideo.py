import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from astropy.cosmology import Planck13

# IMPORTANT NOTE: the redshifts of the snapshots must be taken from the simulation 
# data and saved as a numpy array called 'redshifts.npy'
redshifts = np.load('redshifts.npy')

# cropping & overlaying redshift and lookback time information onto the images
for i in range(0,255):

	# change the file name to be whatever you named the images
    uncropped = Image.open('image'+str(i)+'.png')
    
    # the images can be cropped however you like - just change "area"
    area = (205, 195, 1005, 1005)
    image = uncropped.crop(area)
    draw = ImageDraw.Draw(image)

    # overlays the redshift
    font = ImageFont.truetype('FreeMono.ttf', size=30)
    (x, y) = (20, 20)
    ztext = "z = %.2f" % redshifts[i]
    color = 'rgb(255, 255, 255)'
    draw.text((x, y), ztext, fill=color, font=font)

    # overlays the lookback time
    (x, y) = (20, 50)
    timetext = "Lookback Time = %.2f Gyr" % Planck13.lookback_time(redshifts[i]).value
    draw.text((x, y), timetext, fill=color, font=font)
    
    # saves the new image
    image.save('z_im'+str(i)+'.png')

# saves the images as a video
# NOTE: requires ffmpeg to be installed
os.system(ffmpeg -r 10 -f image2 -s 1920x1080 -i z_im%d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p rotating.mp4