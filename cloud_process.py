# Hank on April 1st, 2021 
# import cloud_process 
## cloud_process for generating patches which needs to consider patch size and patch_step (256 in this code)
from math import pi
from math import cos
import math
import numpy as np
import rasterio
import os
import importlib

import true_color
import color_display

importlib.reload(true_color)

Y_OFFSET = 0
PATCH_SIZE = 512
# PATCH_SIZE = 256
## for central browses 
# PATCH_SIZE = 1024
STARTY_TOA = 3000

## start from 0,0
# PATCH_SIZE = 3072
# STARTX_TOA = 0;
# STARTY_TOA = 0; 

## for spatial mismatch demonstration only
# PATCH_SIZE = 512
# STARTX_TOA = 1024;
# STARTY_TOA = 120; 

## for registration demonstration only
# PATCH_SIZE = 256
# STARTX_TOA = 5000+512;
# STARTY_TOA = 3000+512; 
def save_and_true_color(naip_meta, patch_toa1, browsefile1, patchfile1='', is_DN=False, factor_high=2, factor_low=1):
    if not is_DN and patch_toa1.shape[0]>=3:
        true_color.true_color_from_image_cheatTOA (patch_toa1[1:4,:,:], browsefile1, blue=0, green=1, red=2, 
            factor_high=factor_high, factor_low=factor_low)
    elif patch_toa1.shape[0]>=3:
        # true_color.true_color_from_image (patch_toa1, browsefile1, blue=0, green=1, red=2, b_low=2000, b_high=10000)
        true_color.true_color_from_image (patch_toa1, browsefile1, blue=0, green=1, red=2, b_low=1000, b_high=15000)
    
    if patchfile1!='':
        ## grab spatial information from the input data
        # naip_meta = rasterio.open(dnn_cld_file).profile.copy()
        # Change the count or number of bands from 4 to 1
        naip_meta['count'] = patch_toa1.shape[0] 
        naip_meta['width']  = patch_toa1.shape[2] # a bug fixed on Jul 25 2021 
        naip_meta['height'] = patch_toa1.shape[1] 
        naip_meta['dtype'] = 'int16'
        if patch_toa1.dtype==np.uint16: 
            naip_meta['dtype'] = 'uint16'
        # Write your the ndvi raster object
        with rasterio.open(patchfile1, 'w', **naip_meta) as dst:
            dst.write(patch_toa1)

def save_toa_cld_and_true_color(naip_meta, patch_toa1, patch_cld1, patch_cfmsk1=0, patch_toa_file1='', patch_cld_file1='', patch_cmf_file1='', 
        browse_toa_file1='', browse_cld_file1='', browse_cmf_file1='', factor_high=2, factor_low=1):
    if browse_toa_file1!='':
        true_color.true_color_from_image_cheatTOA (patch_toa1[1:4,:,:], browse_toa_file1, blue=0, green=1, red=2, 
            factor_high=factor_high, factor_low=factor_low)
    
    if browse_cld_file1!='':
        color_display.color_display_from_image(patch_cld1, dsr_file="./cloud_shadow.dsr", output_tif=browse_cld_file1)
    
    if browse_cmf_file1!='':
        color_display.color_display_from_image(patch_cfmsk1, dsr_file="./cloud_shadow.dsr", output_tif=browse_cmf_file1)
    
    if patch_toa_file1!='':
        ## grab spatial information from the input data
        # naip_meta = rasterio.open(dnn_cld_file).profile.copy()
        # Change the count or number of bands from 4 to 1
        naip_meta_temp = naip_meta.copy()
        naip_meta_temp['count'] = patch_toa1.shape[0] 
        naip_meta_temp['width']  = PATCH_SIZE
        naip_meta_temp['height'] = PATCH_SIZE
        naip_meta_temp['dtype'] = 'int16'
        # naip_meta_temp['driver'] = 'JPEG'
        # Write your the ndvi raster object
        with rasterio.open(patch_toa_file1, 'w', **naip_meta_temp) as dst:
            dst.write(patch_toa1)
    
    if patch_cld_file1!='':
        naip_meta_temp = naip_meta.copy()
        naip_meta_temp['count'] = 1
        naip_meta_temp['width']  = PATCH_SIZE
        naip_meta_temp['height'] = PATCH_SIZE
        naip_meta_temp['dtype'] = 'uint8'
        # naip_meta_temp['driver'] = 'JPEG'
        # Write your the ndvi raster object
        with rasterio.open(patch_cld_file1, 'w', **naip_meta_temp) as dst:
            dst.write(patch_cld1, 1)
    
    if patch_cmf_file1!='':
        naip_meta_temp = naip_meta.copy()
        naip_meta_temp['count'] = 1
        naip_meta_temp['width']  = PATCH_SIZE
        naip_meta_temp['height'] = PATCH_SIZE
        naip_meta_temp['dtype'] = 'uint8'
        # naip_meta_temp['driver'] = 'JPEG'
        # Write your the ndvi raster object
        with rasterio.open(patch_cmf_file1, 'w', **naip_meta_temp) as dst:
            dst.write(patch_cfmsk1, 1)

def get_factors(b1_mean, b1_std):
    factor_high=2;        factor_low=1
    if b1_mean>2500*2 or b1_std<20:
        shift_high = 8; shift_low  = 8
        factor_high=(b1_mean+b1_std*shift_high)/2500
        factor_low=(max(b1_mean-b1_std*shift_low,10))/700
    return factor_high,factor_low

##********************************************************************************************************************
## split_images function 
## toa_all: 10 bands TOA, 7 reflective bands, 1 cirrus and 2 thermal bands 
## bqa_toa: toa quality band
## cld_cld: cloud mask
## bqa_cld: cloud quality band used to define no data in cloud mask 
## samples_toa, lines_toa: dimensions of the toa image
## samples_cld, lines_cld: dimensions of the cld mask
## xoffset, yoffset: start points on the TOA images
## line_min_no_fill: minimized pixels = of no-filled values in patches (PATCH_SIZE indicates no fill)
## patch_step: PATCH overlapping 
## file_prefix: used to save data
## ref_profile: profile for geometry 
## mean_in: used to calculate standard deviation

# line_min_no_fill=PATCH_SIZE
# patch_step=PATCH_SIZE
# ref_profile = rasterio.open(dnn_cld_file).profile.copy()
def split_images(toa_all, bqa_toa, cld_cld, bqa_cld, samples_toa, lines_toa, samples_cld, lines_cld, xoffset, yoffset,
    line_min_no_fill=PATCH_SIZE, patch_step=256, PATCH_TOA_DIR='', PATCH_BROWSE_DIR='', file_prefix='', ref_profile='',
    mean_in=0):

    no_filled_toa = np.bitwise_and(bqa_toa,1)==0
    no_filled_cld = np.bitwise_and(bqa_cld,1)==0
    c1_cloud       = np.bitwise_and(np.right_shift(bqa_toa,4),1)==1
    c1_cloud_conf  = np.bitwise_and(np.right_shift(bqa_toa,5),3)>=3 # this is equal to cloud==1
    c1_shadow_conf = np.bitwise_and(np.right_shift(bqa_toa,7),3)>=3
    x_toa_BEGIN = max(0, xoffset)
    y_toa_BEGIN = max(0, yoffset)
    x_toa_END   = min(samples_toa, xoffset+samples_cld)
    y_toa_END   = min(lines_toa  , yoffset+lines_cld  )
    # break;
    ## find y start and y end with at least 512 pixels non-filled both cloud and collection-1
    y_toa_BEGIN_inner = 0; y_toa_END_inner = 0; is_begin_set = False; is_end_set = False
    for iy_toa_start in range(y_toa_BEGIN,y_toa_END):
        filled_sta = np.logical_and (no_filled_cld[0,iy_toa_start-yoffset,(x_toa_BEGIN-xoffset):(x_toa_END-xoffset)],
            no_filled_toa[0,iy_toa_start,x_toa_BEGIN:x_toa_END])
        if filled_sta.sum()>=line_min_no_fill and not is_begin_set:
            y_toa_BEGIN_inner = iy_toa_start
            is_begin_set = True
            break

    for iy_toa_start in range(y_toa_BEGIN_inner,y_toa_END):
        filled_sta = np.logical_and (no_filled_cld[0,iy_toa_start-yoffset,(x_toa_BEGIN-xoffset):(x_toa_END-xoffset)],
            no_filled_toa[0,iy_toa_start,x_toa_BEGIN:x_toa_END])
        if filled_sta.sum()<line_min_no_fill and not is_end_set:
            y_toa_END_inner = iy_toa_start
            is_end_set = True
            break

    if not is_end_set:
        y_toa_END_inner = y_toa_END

    sum_local = 0
    tse_local = 0
    n_local = 0
    # start to process for each 512 * 512 block
    # iy_toa_start = y_toa_BEGIN_inner
    for iy_toa_start in range(y_toa_BEGIN_inner,y_toa_END_inner,patch_step):
        if (iy_toa_start+PATCH_SIZE)>y_toa_END_inner:
            # break
            iy_toa_start = y_toa_END_inner-PATCH_SIZE

        iy_toa_end = iy_toa_start+PATCH_SIZE

        ## find x start and x end with at least 512 pixels non-filled both cloud and collection-1 TOA
        filled_sta = np.logical_and (no_filled_cld[0,iy_toa_start-yoffset,(x_toa_BEGIN-xoffset):(x_toa_END-xoffset)], no_filled_toa[0,iy_toa_start,x_toa_BEGIN:x_toa_END])
        filled_end = np.logical_and (no_filled_cld[0,iy_toa_end-1-yoffset,(x_toa_BEGIN-xoffset):(x_toa_END-xoffset)], no_filled_toa[0,iy_toa_end-1,x_toa_BEGIN:x_toa_END])
        index_sta = np.argwhere(filled_sta); index_sta_left=index_sta[0][0]; index_sta_right=index_sta[-1][0]+1;
        index_end = np.argwhere(filled_end); index_end_left=index_end[0][0]; index_end_right=index_end[-1][0]+1;

        if line_min_no_fill==PATCH_SIZE: ## no filled value
            x_toa_BEGIN_inner = max(index_sta_left ,index_end_left ) + x_toa_BEGIN ## + x_toa_BEGIN must be applied as otherwise it will have a bug
            x_toa_END_inner   = min(index_sta_right,index_end_right) + x_toa_BEGIN ##
        else:
            x_toa_BEGIN_inner = min(index_sta_left ,index_end_left ) + x_toa_BEGIN ## + x_toa_BEGIN must be applied as otherwise it will have a bug
            x_toa_END_inner   = max(index_sta_right,index_end_right) + x_toa_BEGIN ##

        print ("\tiy_toa_start" + str(iy_toa_start) )
        # break
        # ix_toa_start = x_toa_BEGIN_inner
        for ix_toa_start in range(x_toa_BEGIN_inner,x_toa_END_inner,patch_step):
            if (ix_toa_start+PATCH_SIZE)>x_toa_END_inner:
                ix_toa_start = x_toa_END_inner-PATCH_SIZE

            startx_toa = ix_toa_start; starty_toa = iy_toa_start
            # print(ix_toa_start)
            # for ix_toa_start in range
            # a = toa_band01[0, iy_toa_start:iy_toa_end,startx_toa]

            startx_cld = startx_toa-xoffset; starty_cld = starty_toa-yoffset
            # startx_str = str(startx_toa)+'.'+str(starty_toa)
            startx_str = '{:04d}'.format(starty_toa)+'.'+'{:04d}'.format(startx_toa)

            patch_toa_file1  = PATCH_TOA_DIR   +'/'+file_prefix+startx_str+'.TOA.tif'
            patch_cld_file1  = PATCH_TOA_DIR   +'/'+file_prefix+startx_str+'.CLD.tif'
            patch_cmf_file1  = PATCH_TOA_DIR   +'/'+file_prefix+startx_str+'.FMS.tif'
            browse_toa_file1 = PATCH_BROWSE_DIR+'/'+file_prefix+startx_str+'.TOA.tif'
            browse_cld_file1 = PATCH_BROWSE_DIR+'/'+file_prefix+startx_str+'.CLD.tif'
            browse_cmf_file1 = PATCH_BROWSE_DIR+'/'+file_prefix+startx_str+'.FMS.tif'

            # toa_band1[:,starty_toa:(starty_toa+PATCH_SIZE),startx_toa:(startx_toa+PATCH_SIZE)]
            patch_toa1 = toa_all[:,starty_toa:(starty_toa+PATCH_SIZE),startx_toa:(startx_toa+PATCH_SIZE)]
            ## ********************************************************************
            ## start to open cloud patch
            patch_cld1 = cld_cld[:,starty_cld:(starty_cld+PATCH_SIZE),startx_cld:(startx_cld+PATCH_SIZE)].reshape(PATCH_SIZE,PATCH_SIZE)
            patch_cfmsk1 = patch_cld1.copy(); patch_cfmsk1[:,:] = 128
            patch_cfmsk1 [np.logical_or(c1_cloud,c1_cloud_conf)[:,starty_toa:(starty_toa+PATCH_SIZE),
                startx_toa:(startx_toa+PATCH_SIZE)].reshape(PATCH_SIZE,PATCH_SIZE)] = 255
            patch_cfmsk1 [c1_shadow_conf                       [:,starty_toa:(starty_toa+PATCH_SIZE),
                startx_toa:(startx_toa+PATCH_SIZE)].reshape(PATCH_SIZE,PATCH_SIZE)] = 64

            ## ********************************************
            ## find out the intersection
            combined_no_filled = np.logical_and (no_filled_cld[0,starty_cld:(starty_cld+PATCH_SIZE),startx_cld:(startx_cld+PATCH_SIZE)],
                no_filled_toa[:,starty_toa:(starty_toa+PATCH_SIZE),startx_toa:(startx_toa+PATCH_SIZE)]).reshape(PATCH_SIZE,PATCH_SIZE)

            patch_toa1  [:,np.logical_not(combined_no_filled)] = -32768
            patch_cld1  [np.logical_not(combined_no_filled)]   = 0
            patch_cfmsk1[np.logical_not(combined_no_filled)]   = 0
            b1_mean = patch_toa1[0,no_filled_cld[0,starty_cld:(starty_cld+PATCH_SIZE),startx_cld:(startx_cld+PATCH_SIZE)]].mean()
            b1_std =  patch_toa1[0,no_filled_cld[0,starty_cld:(starty_cld+PATCH_SIZE),startx_cld:(startx_cld+PATCH_SIZE)]].std()

            ## for registration demonstration only
            print ('\t\tB1 mean '+'{:4.1f}'.format(b1_mean) + 'B1 std '+'{:4.1f}'.format(b1_std))
            # factor_high=2;        factor_low=1
            # if b1_mean>2500*2 or b1_std<20:
                # shift_high = 8; shift_low  = 8
                # factor_high=(b1_mean+b1_std*shift_high)/2500
                # factor_low=(max(b1_mean-b1_std*shift_low,10))/700
                # print('\tfactor = ' +str(factor_high)+' low factor = '+str(factor_low))
            factor_high,factor_low = get_factors(b1_mean,b1_std)
            naip_meta = ref_profile.copy()
            # rasterio.open(dnn_cld_file).profile.copy()
            if combined_no_filled.sum()<PATCH_SIZE**2 and line_min_no_fill==PATCH_SIZE:
                print ('!!!!!combined_no_filled.sum()<PATCH_SIZE**2 ' + str(combined_no_filled.sum())+'  \t'+ file_prefix+startx_str)
                browse_toa_file1 = PATCH_BROWSE_DIR+'/_inv_'+file_prefix+startx_str+'.TOA.tif'
                browse_cld_file1 = PATCH_BROWSE_DIR+'/_inv_'+file_prefix+startx_str+'.CLD.tif'
                save_toa_cld_and_true_color(naip_meta, patch_toa1, patch_cld1, patch_toa_file1='', patch_cld_file1='', browse_toa_file1=browse_toa_file1,
                    browse_cld_file1=browse_cld_file1, factor_high=factor_high, factor_low=factor_high)
            else:
                save_toa_cld_and_true_color(naip_meta, patch_toa1, patch_cld1, patch_cfmsk1, patch_toa_file1, patch_cld_file1, patch_cmf_file1,
                    browse_toa_file1, browse_cld_file1, browse_cmf_file1, factor_high, factor_low)
                sum_local = sum_local+patch_toa1.sum(axis=(1,2))/PATCH_SIZE**2
                n_local = n_local+1
                if type(mean_in).__module__ == np.__name__:
                    # tse_local = tse_local+ (patch_toa1.reshape(patch_toa1.shape[0],PATCH_SIZE**2)-mean_in)**2/PATCH_SIZE**2
                    tse_local = tse_local+( (patch_toa1-mean_in.reshape(mean_in.shape[0],1,1))**2 ).sum(axis=(1,2))/PATCH_SIZE**2
            # break;
        # break
    return sum_local, tse_local, n_local
