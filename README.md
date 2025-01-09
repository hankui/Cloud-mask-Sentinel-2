# Cloud-and-Shadow-Prediction
 Python implementation for detecting Sentinel-2 clouds and shadows using Swin-Unet model.

## Requirements
- **Programming Languages**: Python 3.7+
- **Libraries**:
  - `tensorflow`
  - `numpy`
  - `rasterio`
## Included files
#### 1. `apply_prediction_v14_3c.py`
This is the main script for applying the trained  model to detect clouds and shadows in multi-band Sentinel-2 data.





 
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
- CNN classification results  (CNN.tif).
- QA images  (CQA.tif).
- Visualization  (CNN3C.browse.tif).

## Citation
Luo, D., Zhang, H. K., Ornelas De Lemos, H., et al., 2025. A global applicable Sentinel-2 cloud and shadow detection model based on Swin Transformer. Science of Remote Sensing. under review.
