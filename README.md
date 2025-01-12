# Cloud and Shadow Detection for Sentinel-2
The python codes for detecting Sentinel-2 clouds and shadows using Swin-Unet model from the Top of the Atmosphere (TOA) reflectance and the codes can be directly applied to the Sentinel-2 images downloaded from ESA. This is ready-to-use codes with the Swin-Unet model already trained using Publicly available annotated cloud and shadow datasets that have been refined and augmented by the authors. The trained model was too huge to be here and was stored in 
https://zenodo.org/records/14630367. 
More details can be found in Luo, D., Zhang, H. K., Ornelas De Lemos, H., et al., 2025. A global applicable Sentinel-2 cloud and shadow detection model based on Swin Transformer. Science of Remote Sensing. under review. 

## Requirements
- **Programming Languages**: Python 3.7+
- **Libraries**:
  - `tensorflow`
  - `numpy`
  - `rasterio`
## Included files
1. `apply_prediction_v14_3c.py` 
-This is the main script for applying the trained model to detect clouds and shadows.
2. `cloud_process.py` 
-This script contains functions for cloud and shadow processing
3. `color_display.py`
-Display, visualize and save images
4. `true_color.py`
-True color display functions
5. `cloud_shadow.dsr`
-ENVI density slice range file
6. `mean.std.no.fill.stl2.v20240930.csv`
-csv file storing the mean and standard deviation values for each band used for normalization

## Usage
`python apply_prediction_v14_3c.py <toa_path> <model_path> <prefix> <output_dir>`
 - toa_path: Path to the directory containing the TOA images.
 - model_path: Path to the trained deep learning model.
 - prefix: Prefix for naming output files.
 - output_dir: Directory to save output files.

Here is an example of <toa_path>
```
T06VWN_toa_path/
├── S2A_MSIL1C_20230319T210201_N0509_R100_T06VWN_20230319T230232.SAFE/
├── S2A_MSIL1C_20230322T211511_N0509_R143_T06VWN_20230322T231220.SAFE/
│   ├── T06VWN_20230322T211511_B01.jp2
│   ├── T06VWN_20230322T211511_B02.jp2
│   ├── ...
│   ├── T06VWN_20230322T211511_B12.jp2
│   └── T06VWN_20230322T211511_TCI.jp2
├── ...
└── S2B_MSIL1C_20231129T210829_N0509_R100_T06VWN_20231129T211618.SAFE
```
## Output
- QA images

| Bit Position | QA Value (Decimal)       | Description                                                                           |
|--------------|----------------------|---------------------------------------------------------------------------------------|
| 0            | 1                    | **Clear Pixel**: The pixel is classified as clear (not cloud, shadow, or fill).       |
| 1            | 2                    | **Cloud Pixel**: The pixel is classified as a cloud.                                  |
| 2            | 4                    | **Edge Pixel**: The pixel is at the edge of a cloud or shadow region.                 |
| 3            | 8                    | **Shadow Pixel**: The pixel is classified as a shadow.                                |
| 8            | 256                  | **Fill Pixel**: The pixel is classified as a fill value (e.g., no valid data).        |


- Visualization images
  - Clear Pixel:    (0,      0,    255)  
  - Cloud Pixel:    (255,    0,      0)
  - Shadow Pixel:   (255,    255,    0)
  - Fill Pixel:     (0,    0,    0)
   

## Citation
Luo, D., Zhang, H. K., Ornelas De Lemos, H., et al., 2025. A global applicable Sentinel-2 cloud and shadow detection model based on Swin Transformer. Science of Remote Sensing. under review.
