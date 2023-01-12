import cv2 as cv
import sys
import os, os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

np.set_printoptions(threshold=sys.maxsize)

gui = True 

directory = os.path.join('.', 'imagens', 'medicao_2mm')
lst = os.listdir(directory)
number_files = len(lst)

# Data from setup
x_step = 2.0 # size of steps between pictures (mm)
h = 49 # distance between camera and laser (mm) ## CONVERT EVERYTHING TO MILIMETERS
rpc = 0.0005 # radians per pitch
ro = 0.2343 # radius offset (compensates for alignment errors)
px_2_mm = int(1200/135) # field of view of the camera (vertical) = total pixels (vertically) / fov vertical (mm) = 135mm
left_slice = 290 # left slice of image ROI

# Constants from code
x_mm = 0.0
df_merged = pd.DataFrame()

for take in range(1, int(number_files)):
    # Read images
    img0 = cv.imread(os.path.join(directory, 'img01.bmp'), cv.IMREAD_GRAYSCALE)
    img1 = cv.imread(os.path.join(directory, f'img{take}0.bmp'), cv.IMREAD_GRAYSCALE)
    if img0 is None or img1 is None:
        sys.exit("Could not read the image.")

    im0 = img0[200:1100,left_slice:]
    im1 = img1[200:1100,left_slice:]

    (height, width) = im0.shape[:2]

    # gamma correction, make values linear
    #im0 = (im0 / np.float32(255)) ** 2.2
    #im1 = (im1 / np.float32(255)) ** 2.2

    diff = cv.absdiff(im1, im0)
    diff = cv.GaussianBlur(diff, ksize=None, sigmaX=3.0)

    # define range of grey color in mono
    lower_value = 40
    upper_value = 256
    # Threshold the mono image to get only laser line
    mask = cv.inRange(diff, lower_value, upper_value)
    # Bitwise-AND mask and original image
    thresh = cv.bitwise_and(diff, diff, mask= mask)
    # diff_nzero = diff.nonzero() # filter non-zero pixels
    indices_vals = np.amax(thresh, axis=1)
    indices = np.argmax(thresh, axis=1) # horizontally, for each row
    null_indices = indices_vals == 0 # filter null indices

    out = cv.cvtColor(thresh.copy(),cv.COLOR_GRAY2BGR)
    # "drawing" 3 pixels thick
    out[np.arange(height), indices-1] = (0,255,0)
    out[np.arange(height), indices  ] = (0,255,0)
    out[np.arange(height), indices+1] = (0,255,0)

    #  Clean empty rows in picture
    for n, value in enumerate(null_indices):
        if value:
            # print(n,0)
            out[n, 0] = (0,0,0)
            out[n, 0-1] = (0,0,0)
            out[n, 0+1] = (0,0,0)

    if take == 4:
        cv.namedWindow("Display window", cv.WINDOW_NORMAL)
        cv.setWindowProperty("Display window", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.imshow('Display window', im1)

        cv.namedWindow("Display window2", cv.WINDOW_NORMAL)
        cv.setWindowProperty("Display window2", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.imshow('Display window2', out)

    # Create dataset with x,y coordinates of the laser and distance from center
    # of the image

    dataset = {'x_displacement': indices}
    df = pd.DataFrame(data=dataset)
    df['y'] = df.index
    df['y_mm'] = df['y'] / px_2_mm
    df['x_displacement'].replace(to_replace=0,  value=490, inplace=True)

    # number of pixels from the center of focal plane
    # dist = x (from sliced image) + 290 (from original image) - im0 width/2 (center of original image)
    df['dist'] = abs(df['x_displacement'] + left_slice - img0.shape[1]/2) 

    # work out distance using D = h/tan(theta)  
    df['theta'] = rpc*df['dist'] + ro
    df['tan_theta'] = np.tan(df['theta'])
    df['z_abs'] = h/(df['tan_theta']) 
    df['z_rel'] = max(df['z_abs']) - df['z_abs']


    # x position from world
    df['x'] = x_mm
    x_mm += x_step

    df_merged = pd.concat([df_merged, df], ignore_index=True, sort=False)

print(df_merged)

# plot surface
fig = px.scatter_3d(df_merged, x='x', y='y_mm', z='z_rel', color='z_rel',
                    color_continuous_scale=px.colors.sequential.Jet)
fig.update_layout(scene_aspectmode='data', title_text="Laser Triangulation Mapping",
                title_x=0.5)
fig.update_scenes(xaxis_title='Eixo x (mm)', yaxis_title='Eixo y (mm)', zaxis_title='Eixo z (mm)')
fig.show()

k = cv.waitKey(0)
if k == ord("q"):
    cv.destroyAllWindows()
    sys.exit("Exit")
