# true colour display 
# Hankui Jul 11 2020 
# import true_color

import os
import subprocess
import datetime
import rasterio

## ************************************************************************
## convert from tif to ENVI
stretch_cmd = "truecolor.log.asTIFF"

WORK_DIR = "./"
if not os.path.exists('./tmp'):
    os.makedirs('./tmp')


## new stretch after Andy's 2000 epoch
b_low=200
b_high=2800

## Landsat-8 stretch
b_low=148
b_high=4915

## ABI stretch
# b_low=300
# b_high=10000

## Landsat-8 DN stretch
# b_low=2000
# b_high=10000

def true_color (input_file, win=0, blue=0, green=1, red=2):
    prefix = str(datetime.datetime.now().second) + str(datetime.datetime.now().minute) + str(datetime.datetime.now().hour)
    filenames = [WORK_DIR+"tmp/"+prefix+".blue", WORK_DIR+"tmp/"+prefix+".green", WORK_DIR+"tmp/"+prefix+".red"]
    base1 = os.path.basename(input_file)
    ture_color_tif_name = WORK_DIR+"True_color."+base1
    
    ## ************************************************************************
    ## convert from tif to ENVI and tif image
    with rasterio.open(input_file) as src:
        # print(src1.profile)
        if win==0:
            win = rasterio.windows.Window(0, 0, src.width, src.height)
        
        image = src.read(window=win)
        dims = image.shape
        
        image[blue ,:,:].tofile(filenames[0])
        image[green,:,:].tofile(filenames[1])
        image[red  ,:,:].tofile(filenames[2])
        
        nrow = dims[1]
        ncol = dims[2]
        subprocess.run([stretch_cmd, str(nrow), str(ncol), filenames[0], str(b_low), str(b_high), 
            filenames[1], str(b_low), str(b_high), filenames[2], str(b_low), str(b_high), ture_color_tif_name ])
    
    ## ************************************************************************
    ## delete temporary files 
    if os.path.isfile(filenames[0]): 
        os.remove(filenames[0])
    if os.path.isfile(filenames[1]): 
        os.remove(filenames[1])
    if os.path.isfile(filenames[2]): 
        os.remove(filenames[2])


## numbers copied from d:\mycode\mycode.wylde\mk.jpg\truecolor.jpg.sh
b1_low=700
b1_high=2500

b2_low=500
b2_high=2300

b3_low=400
b3_high=2200

## numbers made up by Hank for Landsat-8 according to above 
# b1_low=700
# b1_high=2500*2

# b2_low=500
# b2_high=2300*2

# b3_low=400
# b3_high=2200*2

def true_color_from_image_cheatTOA (image, ture_color_tif_name, blue=0, green=1, red=2,factor_high=2, factor_low=1 ):
    prefix = str(datetime.datetime.now().second) + str(datetime.datetime.now().minute) + str(datetime.datetime.now().hour)
    filenames = [WORK_DIR+"tmp/"+prefix+".blue", WORK_DIR+"tmp/"+prefix+".green", WORK_DIR+"tmp/"+prefix+".red"]
    # base1 = os.path.basename(input_file)
    # ture_color_tif_name = WORK_DIR+"True_color."+base1
    
    ## ************************************************************************
    ## convert from tif to ENVI and tif image
    # with rasterio.open(input_file) as src:
        # print(src1.profile)
        # if win==0:
            # win = rasterio.windows.Window(0, 0, src.width, src.height)
        
        # image = src.read(window=win)
    dims = image.shape
    
    image[blue ,:,:].tofile(filenames[0])
    image[green,:,:].tofile(filenames[1])
    image[red  ,:,:].tofile(filenames[2])
    
    nrow = dims[1]
    ncol = dims[2]
    subprocess.run([stretch_cmd, str(nrow), str(ncol), filenames[0], str(int(b1_low*factor_low)), str(int(b1_high*factor_high)), 
        filenames[1], str(int(b2_low*factor_low)), str(int(b2_high*factor_high)), filenames[2], str((b3_low*factor_low)), str(int(b3_high*factor_high)), ture_color_tif_name ])
    
    ## ************************************************************************
    ## delete temporary files 
    if os.path.isfile(filenames[0]): 
        os.remove(filenames[0])
    if os.path.isfile(filenames[1]): 
        os.remove(filenames[1])
    if os.path.isfile(filenames[2]): 
        os.remove(filenames[2])

def true_color_from_image (image, ture_color_tif_name, blue=0, green=1, red=2, b_low=b_low, b_high=b_high):
    prefix = str(datetime.datetime.now().second) + str(datetime.datetime.now().minute) + str(datetime.datetime.now().hour)
    filenames = [WORK_DIR+"tmp/"+prefix+".blue", WORK_DIR+"tmp/"+prefix+".green", WORK_DIR+"tmp/"+prefix+".red"]
    # base1 = os.path.basename(input_file)
    # ture_color_tif_name = WORK_DIR+"True_color."+base1
    
    ## ************************************************************************
    ## convert from tif to ENVI and tif image
    # with rasterio.open(input_file) as src:
        # print(src1.profile)
        # if win==0:
            # win = rasterio.windows.Window(0, 0, src.width, src.height)
        
        # image = src.read(window=win)
    dims = image.shape
    
    image[blue ,:,:].tofile(filenames[0])
    image[green,:,:].tofile(filenames[1])
    image[red  ,:,:].tofile(filenames[2])
    
    nrow = dims[1]
    ncol = dims[2]
    subprocess.run([stretch_cmd, str(nrow), str(ncol), filenames[0], str(b_low), str(b_high), 
        filenames[1], str(b_low), str(b_high), filenames[2], str(b_low), str(b_high), ture_color_tif_name ])
    
    ## ************************************************************************
    ## delete temporary files 
    if os.path.isfile(filenames[0]): 
        os.remove(filenames[0])
    if os.path.isfile(filenames[1]): 
        os.remove(filenames[1])
    if os.path.isfile(filenames[2]): 
        os.remove(filenames[2])


def true_color_for_tiles(min_scatter,scatteri,intiles,postfix=".truecolor.tif"):
    import numpy as np
    import math 
    import ABI_MAIAC_multi_tile
    which_min = np.where(min_scatter==scatteri)[0][0]
    temp_ref = intiles.reflectances_1km[which_min,:,:,:]*10000
    ture_color_tif_name = "./"+'doy.{:03d}'.format(intiles.doys[which_min])+'.{:04d}'.format(intiles.times[which_min])+'.scatter{:03d}'.format(int(scatteri))+'.local{:02d}'.format(intiles.local_hours[which_min])+'.{:02d}'.format(intiles.local_minutes[which_min])+postfix
    centeri = int(ABI_MAIAC_multi_tile.DIM_5KM/2)
    centerj = int(ABI_MAIAC_multi_tile.DIM_5KM/2)
    print("sz centre = " + str(np.arccos(intiles.cosszs[which_min, centeri,centerj])*180/math.pi) )
    print("sc centre = " + str(          intiles.scas  [which_min, centeri,centerj]) )
    print(intiles.base_names[which_min])
    temp_image = np.full([3,ABI_MAIAC_multi_tile.DIM_1KM, ABI_MAIAC_multi_tile.DIM_1KM], fill_value=ABI_MAIAC_multi_tile.fill_value, dtype=np.int16) ## must use fill
    temp_image[0,:,:] = (temp_ref[0,:,:]).astype(np.int16)
    temp_image[1,:,:] = (0.45*temp_ref[0,:,:]+0.45*temp_ref[1,:,:]+0.1*temp_ref[2,:,:]+0.5).astype(np.int16)
    temp_image[2,:,:] = (temp_ref[1,:,:]).astype(np.int16)
    true_color_from_image(temp_image, ture_color_tif_name)