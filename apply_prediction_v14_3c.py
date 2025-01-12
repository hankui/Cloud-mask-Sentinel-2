# v14_3c July 12, 2024 the input is all 13 bands, using 3 classes model, speed up predict, result: CQA and browse

import sys
import os
import gc
import numpy as np
import rasterio
from rasterio.enums import Resampling
import tensorflow as tf
import cloud_process
import importlib
importlib.reload (cloud_process)

## load mean and stardard varition
mean_name='./mean.std.no.fill.stl2.v20240930.csv'
x_mean = 0
x_std = 1
if os.path.exists(mean_name):
    dat_temp = np.loadtxt(open(mean_name, "rb"), dtype='<U30', delimiter=",", skiprows=1)
    arr = dat_temp.astype(np.float64)
    # x_dim = arr.shape[0]-1
    x_mean,x_std = arr[:,0],arr[:,1]
else:
    print("Error !!!! mean file not exists " + mean_name)

## parameters
IMG_BANDS = 13
PATCH_SIZE = cloud_process.PATCH_SIZE
UNIQUE_LABELS = [64, 128, 255]

##**************************************************************************************************
# QA function
def set_bit (value, bit_index):
    return value | (1 << bit_index)

def img_to_qa (parr):
    """
    function to set qa bit by just set each pixel as a single bit
    bug fix: replace np.insert to np.hstack or np.vstack when add array at the end of array
    """    
    cldmsk = parr == 255
    csmsk = parr == 64
    fillmsk = parr == 10

    addc_arr = np.zeros(parr.shape[0], dtype = np.int16)
    addr_arr = np.zeros(parr.shape[1], dtype = np.int16)

    left_arr1 = np.vstack((np.delete(parr, 0, axis=0),  addr_arr))    
    left_up_arr1 = np.hstack((np.delete(left_arr1, 0, axis=1), np.expand_dims(addc_arr, 1)))   
    left_down_arr1 = np.insert(np.delete(left_arr1, -1, axis=1), 0, addc_arr, axis=1)
    rigt_arr1 = np.insert(np.delete(parr,-1, axis=0), 0, addr_arr, axis=0)    
    rigt_up_arr1 = np.hstack((np.delete(rigt_arr1, 0, axis=1), np.expand_dims(addc_arr, 1)))    
    rigt_down_arr1 = np.insert(np.delete(rigt_arr1, -1, axis=1), 0, addc_arr, axis=1)
    up_arr1 = np.hstack((np.delete(parr, 0, axis=1), np.expand_dims(addc_arr, 1)))
    down_arr1 = np.insert(np.delete(parr,-1, axis=1), 0, addc_arr, axis=1)
    edge_cmsk = (left_arr1==255)|(rigt_arr1==255)|(up_arr1==255)|(down_arr1==255)|(left_up_arr1==255)|(left_down_arr1==255)|(rigt_up_arr1==255)|(rigt_down_arr1==255)
    edge_cmsk[cldmsk] = 0    
    edge_clds = (left_arr1==64)|(rigt_arr1==64)|(up_arr1==64)|(down_arr1==64)|(left_up_arr1==64)|(left_down_arr1==64)|(rigt_up_arr1==64)|(rigt_down_arr1==64) 
    edge_clds[csmsk] = 0    
    clearmsk = (np.logical_not(cldmsk)) & (np.logical_not(csmsk)) & (np.logical_not(fillmsk)) & (np.logical_not(edge_cmsk)) & (np.logical_not(edge_clds)) 
    qaarr = np.zeros((parr.shape[0], parr.shape[1]), dtype = np.int16)

    qaarr[cldmsk]                                                  = set_bit(0b0000000000000000, 1) 
    qaarr[csmsk]                                                   = set_bit(0b0000000000000000, 3) 
    qaarr[clearmsk]                                                = set_bit(0b0000000000000000, 0)     
    qaarr[edge_cmsk & np.logical_not(np.logical_or(cldmsk,csmsk))] = set_bit(0b0000000000000000, 2)           
    qaarr[edge_clds & np.logical_not(np.logical_or(cldmsk,csmsk))] = set_bit(0b0000000000000000, 2)
    qaarr[fillmsk]                                                 = set_bit(0b0000000000000000, 8)
    return qaarr
        
def img_to_qa0 (parr):
    """function to set qa bit by just set each pixel as a single bit"""    
    cldmsk = parr == 255
    csmsk = parr == 64
    fillmsk = parr == 10
    addc_arr = np.zeros(parr.shape[0], dtype = np.int16)
    addr_arr = np.zeros(parr.shape[1], dtype = np.int16)
    left_arr1 = np.insert(np.delete(parr, 0, axis=0), -1, addr_arr, axis=0)    
    left_up_arr1 = np.insert(np.delete(left_arr1, 0, axis=1), -1, addc_arr, axis=1)    
    left_down_arr1 = np.insert(np.delete(left_arr1, -1, axis=1), 0, addc_arr, axis=1)
    rigt_arr1 = np.insert(np.delete(parr,-1, axis=0), 0, addr_arr, axis=0)   
    rigt_up_arr1 = np.insert(np.delete(rigt_arr1, 0, axis=1), -1, addc_arr, axis=1)    
    rigt_down_arr1 = np.insert(np.delete(rigt_arr1, -1, axis=1), 0, addc_arr, axis=1)
    up_arr1 = np.insert(np.delete(parr, 0, axis=1), -1, addc_arr, axis=1)
    down_arr1 = np.insert(np.delete(parr,-1, axis=1), 0, addc_arr, axis=1)
    edge_cmsk = (left_arr1==255)|(rigt_arr1==255)|(up_arr1==255)|(down_arr1==255)|(left_up_arr1==255)|(left_down_arr1==255)|(rigt_up_arr1==255)|(rigt_down_arr1==255)
    edge_cmsk[cldmsk] = 0   
    edge_clds = (left_arr1==64)|(rigt_arr1==64)|(up_arr1==64)|(down_arr1==64)|(left_up_arr1==64)|(left_down_arr1==64)|(rigt_up_arr1==64)|(rigt_down_arr1==64) 
    edge_clds[csmsk] = 0    
    clearmsk = (np.logical_not(cldmsk)) & (np.logical_not(csmsk)) & (np.logical_not(fillmsk)) & (np.logical_not(edge_cmsk)) & (np.logical_not(edge_clds)) 
    qaarr = np.zeros((parr.shape[0], parr.shape[1]), dtype = np.int16)
    qaarr[cldmsk]                                                  = set_bit(0b0000000000000000, 1) 
    qaarr[csmsk]                                                   = set_bit(0b0000000000000000, 3) 
    qaarr[clearmsk]                                                = set_bit(0b0000000000000000, 0)     
    qaarr[edge_cmsk & np.logical_not(np.logical_or(cldmsk,csmsk))] = set_bit(0b0000000000000000, 2)           
    qaarr[edge_clds & np.logical_not(np.logical_or(cldmsk,csmsk))] = set_bit(0b0000000000000000, 2)
    qaarr[fillmsk]                                                 = set_bit(0b0000000000000000, 8)
    return qaarr

def cloud_20mto30m(cmask):
    ## initialize 
    shape30m = (int(cmask.shape[0]*2/3),int(cmask.shape[1]*2/3))
    cmask30m = np.full(shape30m,fill_value=128,dtype=cmask.dtype)    
    ## get masks 
    filled_mask = cmask==10
    cloud_mask = cmask==255
    shadow_mask = cmask==64
    clear_mask = cmask==128    
    ## get masks 
    x = np.array(range(shape30m[0]))
    y = np.array(range(shape30m[1]))
    x20m_0 = np.floor(x/2*3).astype(int)
    x20m_1 = np.floor((x+0.99)/2*3).astype(int)
    y20m_0 = np.floor(y/2*3).astype(int)
    y20m_1 = np.floor((y+0.99)/2*3).astype(int)
    xx0, yy0 = np.meshgrid(x20m_0, y20m_0)
    xx0, yy1 = np.meshgrid(x20m_0, y20m_1)
    xx1, yy0 = np.meshgrid(x20m_1, y20m_0)
    xx1, yy1 = np.meshgrid(x20m_1, y20m_1)
    cloud_mask20m  = np.logical_or.reduce(( cloud_mask[yy0, xx0], cloud_mask[yy0, xx1], cloud_mask[yy1, xx0], cloud_mask[yy1, xx1]))
    shadow_mask20m = np.logical_or.reduce((shadow_mask[yy0, xx0],shadow_mask[yy0, xx1],shadow_mask[yy1, xx0],shadow_mask[yy1, xx1]))
    filled_mask20m = np.logical_or.reduce((filled_mask[yy0, xx0],filled_mask[yy0, xx1],filled_mask[yy1, xx0],filled_mask[yy1, xx1]))    
    ## set masks 
    cmask30m[shadow_mask20m] = 64
    cmask30m[cloud_mask20m ] = 255
    cmask30m[filled_mask20m] = 10    
    return cmask30m

# function for resample 10m to 20m
def average10_20 (image10m):
   xsize10 = image10m.shape[1] 
   ysize10 = image10m.shape[2]
   
   xsize20 =  int (xsize10/2)
   ysize20 =  int (ysize10/2)
   
   image10m = image10m.reshape(xsize10, ysize10)
   col_index = np.repeat(np.array(range(0,xsize10,2)).reshape(1,xsize20),ysize20, axis=0).reshape(xsize20*ysize20)
   row_index = np.repeat(np.array(range(0,ysize10,2)),xsize20)
   topleft_index = row_index*xsize10+col_index
   topright_index = topleft_index+1
   bottom_left_index = topleft_index+xsize10
   bottom_right_index = topleft_index+xsize10+1
   image10m1d = image10m.reshape(xsize10*ysize10)
   mask1d = image10m1d != 0 
   sum1d = mask1d[topleft_index].astype(np.int8)+mask1d[bottom_right_index].astype(np.int8)+mask1d[bottom_left_index].astype(np.int8)+mask1d[topright_index].astype(np.int8)
   # image20m = ((image10m1d[topleft_index]+image10m1d[bottom_right_index]+image10m1d[bottom_left_index]+image10m1d[topright_index])/4+0.5).astype(np.int16).reshape(1, xsize20, ysize20)
   image20m = (np.divide(image10m1d[topleft_index]+image10m1d[bottom_right_index]+image10m1d[bottom_left_index]+image10m1d[topright_index], sum1d,where=sum1d>0)+0.5).astype(np.int16).reshape(1, xsize20, ysize20)
   return image20m

def find_one_file (dir1, pattern=''):
    find_file = ''
    for root, dirs, files in os.walk(dir1):
        for file in files:            
            if pattern in file:            
                find_file = os.path.join(root, file)    
    return find_file
    
def covert_123_to_label(fms_image):
    fms_image_out = fms_image.copy()
    for ci in range(len(UNIQUE_LABELS)):
        fms_image_out[fms_image==ci] = UNIQUE_LABELS[ci]
    return fms_image_out

##********************************************************************************************************************
## This function is for generating patches
## xoffset, yoffset: start points on the TOA images
## file_prefix: used to save data
## ref_profile: profile for geometry
def split_toa_for_prediction (toa_all, bqa_toa, samples_toa, lines_toa,                               
                              patch_size=512, is_norm=True,
                              line_min_no_fill=1, patch_step=PATCH_SIZE):
    '''
    This function is for generating patches
    Args:
        toa_all: 10 bands TOA, 7 reflective bands, 1 cirrus and 2 thermal bands
        bqa_toa: toa quality band
        samples_toa: dimensions of the toa image
        lines_toa: dimensions of the toa image
        patch_size:
        is_norm:
        line_min_no_fill: minimized pixels = of no-filled values in patches (PATCH_SIZE indicates no fill)
        patch_step: PATCH overlapping
    Returns:
    '''
    no_filled_toa = bqa_toa!=0
    print(no_filled_toa.shape)
    x_toa_BEGIN = 0
    y_toa_BEGIN = 0
    x_toa_END   = samples_toa
    y_toa_END   = lines_toa
    for iy_toa_start in range(y_toa_BEGIN,y_toa_END):
        filled_sta = no_filled_toa[0,iy_toa_start,x_toa_BEGIN:x_toa_END]
        for bi in range(IMG_BANDS):
            # fixed a bug on Jul 24 as each band has different widths (thermal & reflective)
            index_sta = np.argwhere(np.logical_and(filled_sta, toa_all[bi,iy_toa_start,x_toa_BEGIN:x_toa_END]>0) ); 
            if index_sta.sum() == 0:
                break
            index_sta_left=index_sta[0][0]; index_sta_right=index_sta[-1][0]+1; 
            # valid_len = (index_sta_right-index_sta_left)
            toa_all[bi,iy_toa_start, np.logical_not(filled_sta)] = toa_all[bi,iy_toa_start,filled_sta].mean()
            if index_sta_left>0:
                toa_all[bi,iy_toa_start,0:(index_sta_left)] = toa_all[bi,iy_toa_start,index_sta_left]
            if index_sta_right<x_toa_END:
                toa_all[bi,iy_toa_start,index_sta_right:x_toa_END] = toa_all[bi,iy_toa_start,(index_sta_right-1)]
    ## *************************************************************************
    ## define return variables 
    sum_local = 0
    # tse_local = 0
    n_local = 0
    MAX_PATCHES = 400
    TEST_x = np.full([MAX_PATCHES, patch_size, patch_size, IMG_BANDS], fill_value=-9999, dtype=np.float32)   
    START_x = list()
    START_y = list()
    is_top  = list()
    is_left = list()
    ## *************************************************************************
    # start to process for each 512 * 512 block
    # iy_toa_start = y_toa_BEGIN_inner
    for iy_toa_start in range(y_toa_BEGIN,y_toa_END,patch_step):
        if (iy_toa_start+patch_size)>y_toa_END: ## last block move back to fit patch size
            iy_toa_start = y_toa_END-patch_size
        print ("\tiy_toa_start" + str(iy_toa_start) )
        print ("\t\tx_toa_BEGIN_inner" + str(x_toa_BEGIN)+"\tx_toa_END_inner" + str(x_toa_END) )
        for ix_toa_start in range(x_toa_BEGIN,x_toa_END,patch_step):
            if (ix_toa_start+patch_size)>x_toa_END: ## last block move back to fit patch size
                ix_toa_start = x_toa_END-patch_size                
            startx_toa = ix_toa_start; starty_toa = iy_toa_start
            startx_str = '{:04d}'.format(starty_toa)+'.'+'{:04d}'.format(startx_toa)                                         
            patch_toa1 = toa_all[:,starty_toa:(starty_toa+patch_size),startx_toa:(startx_toa+patch_size)]    # in for loop that shape is (10, 512, 512)
            for bi in range(IMG_BANDS):
                if is_norm:
                    TEST_x[n_local,:,:,bi] = (patch_toa1[bi,:,:].astype(np.float32)-x_mean[bi])/x_std[bi]
                else:
                    TEST_x[n_local,:,:,bi] = patch_toa1[bi,:,:].astype(np.float32)                           
            ## ********************************************
            ## find out the intersection                        
            combined_no_filled = no_filled_toa[:,starty_toa:(starty_toa+patch_size),startx_toa:(startx_toa+patch_size)].reshape(patch_size,patch_size)   # shape (512, 512)
            b1_mean = patch_toa1[0,combined_no_filled].mean()
            b1_std =  patch_toa1[0,combined_no_filled].std()             
            ## for registration demonstration only
            print ('\t\tB1 mean '+'{:4.1f}'.format(b1_mean) + 'B1 std '+'{:4.1f}'.format(b1_std))            
            # rasterio.open(dnn_cld_file).profile.copy()
            if combined_no_filled.sum()<patch_size**2 and line_min_no_fill==patch_size:
                print ('!!!!!combined_no_filled.sum()<PATCH_SIZE**2 ' + str(combined_no_filled.sum()))            
            # factor_high,factor_low = cloud_process.get_factors(b1_mean,b1_std)
            # naip_meta = ref_profile.copy()
            # cloud_process.save_and_true_color(naip_meta, patch_toa1, browse_toa_file1, patch_toa_file1, factor_high=factor_high, factor_low=factor_low)
            sum_local = sum_local+patch_toa1.sum(axis=(1,2))/patch_size**2
            n_local = n_local+1      # n_local = n_test
            START_x.append(startx_toa)
            START_y.append(starty_toa)
            is_top .append(iy_toa_start==y_toa_BEGIN)
            is_left.append(ix_toa_start==x_toa_BEGIN)
    return TEST_x, n_local, START_x, START_y, is_top, is_left


def get_prediction(model, TEST_x, samples_toa, lines_toa, n_test, START_x, START_y, is_top, is_left, BATCH_SIZE=14, patch_step=512, IMG_HEIGHT=512, IMG_WIDTH=512):
    classes = np.full([lines_toa, samples_toa], fill_value=10, dtype=np.uint8)
    ## 10 is filled in 123 and in label
    tempx = np.full([BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_BANDS], fill_value=-9999, dtype=np.float32)        
    # prediction
    tempi = 0
    starti = 0
    for i in range(n_test):
        tempx[tempi,:,:,:] = TEST_x[i,:,:,:]
        tempi=tempi+1
        if tempi>=(BATCH_SIZE) or i >=(n_test -1):
            print ("process patches "+str(starti+1)+"..."+str(i+1) + "\ttempi = " + str(tempi) )                     
            logits = model.predict(tempx)
            # prop = tf.nn.softmax(logits).numpy()   # v8_1 version
            prop = logits      # v8_3 version
            classesi = np.argmax(prop,axis=3).astype(np.uint8)
            for j in range(tempi):
                jin = j+starti
                if jin>(n_test-1):
                    print ("! number is 1 is invoked") 
                    jin = n_test-1                
                starty_at_j =  START_y[jin] if is_top [jin] else START_y[jin]+int((IMG_HEIGHT-patch_step)/2)
                startx_at_j =  START_x[jin] if is_left[jin] else START_x[jin]+int((IMG_HEIGHT-patch_step)/2)                
                classes[starty_at_j:(START_y[jin]+IMG_HEIGHT),startx_at_j:(START_x[jin]+IMG_WIDTH)] = classesi[j,(starty_at_j-START_y[jin]):IMG_HEIGHT,(startx_at_j-START_x[jin]):IMG_WIDTH]                                                  
            starti = i+1
            tempi = 0
    return classes


# fucntion of predicting new data with trained model 
def predict_to_use(toa_all_file, up_factor, model, BATCH_SIZE=32, IMG_HEIGHT=512, IMG_WIDTH=512):    # toa_bqa_file,
    ## loading data
    ## process 69m bands
    imgb01 = find_one_file(toa_all_file, '_B01.jp2')  
    dataset = rasterio.open(imgb01)                      
    toa_band01 = dataset.read(out_shape =(dataset.count, 
                                        int(dataset.shape[0]*up_factor),
                                        int(dataset.shape[1]*up_factor)),
                                          resampling=Resampling.nearest)          
    toa_band01 = toa_band01
    imgb09 = find_one_file(toa_all_file, '_B09.jp2')  
    dataset = rasterio.open(imgb09)                      
    toa_band09 = dataset.read(out_shape =(dataset.count, 
                                        int(dataset.shape[0]*up_factor),
                                        int(dataset.shape[1]*up_factor)),
                                          resampling=Resampling.nearest)          
    toa_band09 = toa_band09
    imgb10 = find_one_file(toa_all_file, '_B10.jp2')          
    dataset = rasterio.open(imgb10)
    toa_band10 = dataset.read(out_shape =(dataset.count, 
                                                      int(dataset.shape[0]*up_factor),
                                                      int(dataset.shape[1]*up_factor)),
                                          resampling=Resampling.nearest)             
    toa_band10 = toa_band10
    ## process 10m bands
    imgb02 = find_one_file(toa_all_file, '_B02.jp2')     
    dataset = rasterio.open(imgb02)
    toa_band02 = average10_20(dataset.read())  
    toa_band02 = toa_band02

    imgb03 = find_one_file(toa_all_file, '_B03.jp2')                  
    dataset = rasterio.open(imgb03)
    toa_band03 = average10_20(dataset.read())
    toa_band03 = toa_band03  

    imgb04 = find_one_file(toa_all_file, '_B04.jp2')       
    dataset = rasterio.open(imgb04)
    toa_band04 = average10_20(dataset.read())
    toa_band04 = toa_band04     

    imgb08 = find_one_file(toa_all_file, '_B08.jp2')       
    dataset = rasterio.open(imgb08)
    toa_band08 = average10_20(dataset.read())
    toa_band08 = toa_band08
    ## process 10m bands
    imgb8a = find_one_file(toa_all_file, '_B8A.jp2')                                      
    base_name = os.path.basename(imgb8a)[0:40] 
    dataset = rasterio.open(imgb8a)
    toa_band8a0 = dataset.read()
    toa_band8a = toa_band8a0         
    ref_profile = dataset.profile        
    imgb11 = find_one_file(toa_all_file, '_B11.jp2')    
    toa_band11 = rasterio.open(imgb11).read()
    toa_band11 = toa_band11    
    imgb12 = find_one_file(toa_all_file, '_B12.jp2') 
    toa_band12 = rasterio.open(imgb12).read()
    toa_band12 = toa_band12
    
    imgb05 = find_one_file(toa_all_file, '_B05.jp2') 
    toa_band05 = rasterio.open(imgb05).read()
    toa_band05 = toa_band05 
    imgb06 = find_one_file(toa_all_file, '_B06.jp2') 
    toa_band06 = rasterio.open(imgb06).read()
    toa_band06 = toa_band06
    imgb07 = find_one_file(toa_all_file, '_B07.jp2') 
    toa_band07 = rasterio.open(imgb07).read()
    toa_band07 = toa_band07
                                                                       
    toa_all = np.stack([np.squeeze(toa_band01),np.squeeze(toa_band02),np.squeeze(toa_band03),np.squeeze(toa_band04),\
                        np.squeeze(toa_band8a),np.squeeze(toa_band11),np.squeeze(toa_band12),np.squeeze(toa_band10),\
                        np.squeeze(toa_band05),np.squeeze(toa_band06),np.squeeze(toa_band07),np.squeeze(toa_band08),\
                        np.squeeze(toa_band09)], axis = 0).astype(np.int16)

    if '_N05' in toa_all_file:
        toa_all = toa_all - 1000
    else:
        toa_all = toa_all    
    bqa_toa = toa_band8a0
    toa_for_display = toa_all[1:4,:,:].copy() ## as after split the edges are filled 
    ##*********************************************
    ## split x and y 
    samples_toa = toa_all.shape[2]
    lines_toa   = toa_all.shape[1]
    patch_step = 512
    # function 'split_toa_for_prediction' dealing with data and generate patches
    TEST_x, n_test, START_x, START_y, is_top, is_left = split_toa_for_prediction (toa_all, bqa_toa, samples_toa, lines_toa, 
                                                                                  line_min_no_fill=1, patch_step=patch_step)                          
    no_filled_toa = (bqa_toa!=0).reshape(lines_toa, samples_toa)
    b1_mean = toa_all[0,no_filled_toa].mean()
    b1_std  = toa_all[0,no_filled_toa].std()
    factor_high,factor_low = cloud_process.get_factors(b1_mean,b1_std)
    ##*********************************************
    ## get and save prediction 
    classes = get_prediction(model, TEST_x, samples_toa, lines_toa, n_test, START_x, START_y, is_top, is_left, patch_step=patch_step)    # function 'get_prediction'   ,temp_band1
#    classes123 = prepare_data.combine_thin_thick_123(classes)       # (control 3 classes or 4 classes)    
    predicted = covert_123_to_label(classes.astype(np.uint8))
    predicted[np.logical_not(no_filled_toa)] = 10      # this step is for filling the edge patches as "10"
    del toa_all, TEST_x, 
    gc.collect()
    return predicted, ref_profile, toa_for_display, factor_high,factor_low
#*******************************END OF PREDICTION**************************************************

if __name__ == "__main__":   
    toa_path = sys.argv[1]
    model_path = sys.argv[2]    
    prefix = sys.argv[3]
    OUT_DIR  = sys.argv[4]
    OUT_DIR  = os.path.join(OUT_DIR, prefix)
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    model = tf.keras.models.load_model(model_path, compile= False)
    ###################################################################################################
    folders = os.listdir(toa_path)
    total_n = len(folders)
    for i in range(total_n):
        folderi = folders[i]       
        toa_all_file = toa_path + '/' + folderi  
        up_factor = 3    
        cmask, ref_profile, toa20m_rgb, factor_high,factor_low = predict_to_use(toa_all_file, up_factor, model, BATCH_SIZE=32, IMG_HEIGHT=512, IMG_WIDTH=512)   # toa_bqa_file,
        qaimg = img_to_qa(cmask)
        base_name = os.path.basename(os.path.dirname(toa_all_file))[0:60]    # for test purpose
        base_name = os.path.basename(toa_all_file)[0:60]     # for bash run purpose
        import color_display

        cnn_cld_filename = OUT_DIR+'/'+base_name+"."+"CNN."+prefix+".tif"    
        cnn_qa_filename = OUT_DIR+'/'+base_name+"."+"CQA." +prefix+".tif"
        cnn_brw_filename = OUT_DIR+'/'+base_name+"."+"CNN3C.browse." +prefix+".tif"
        # toa_brw_filename = OUT_DIR+'/'+base_name+".TOA.20m.browse.tif"    
        ##********************************************************************
        ## spatial resolution is 30m 
        # true_color.true_color_from_image_cheatTOA (toa20m_rgb, toa_brw_filename, blue=0, green=1, red=2, factor_high=factor_high, factor_low=factor_low)
        color_display.color_display_from_image(cmask,dsr_file="./cloud_shadow.dsr", output_tif=cnn_brw_filename)   
        
        naip_meta = ref_profile.copy()
        naip_meta['driver'] = 'GTiff'       # driver = 'JP2OpenJPEG' [jp2] or 'GTiff' [tif]. tiff file is better 09-03-2021
        naip_meta['count'] = 1
        naip_meta['width']  = cmask.shape[1]   
        naip_meta['height'] = cmask.shape[0] 
    #    transorm_t = naip_meta['transform']
    #    naip_meta['transform'] = rasterio.transform.Affine(30,0,transorm_t[2],0,-30,transorm_t[5])
    #    naip_meta['dtype'] = 'uint8'
    #    with rasterio.open(cnn_cld_filename, 'w', **naip_meta) as dst:
    #        dst.write(cmask30m, 1)
        naip_meta['dtype'] = 'uint16'
        with rasterio.open(cnn_qa_filename, 'w', **naip_meta) as dst:    
            dst.write(qaimg, 1)

#********************************END OF APPLICATION**********************************************************
    
    

