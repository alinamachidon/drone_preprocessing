# Summary of the "analyseDTMs.py" script:
## 1. Loads and organizes all raster files
Reads all .tif files in the target directory

Parses filenames to extract:

date (e.g. 20250205)

location (e.g. BacaPriModreju)

session (e.g. ESAprojekt4)

Organizes files by location and date

## 2. Processes one location at a time
For each location:

Collects all rasters across sessions

Loads profiles and computes the common bounding box (spatial overlap)

## 3. Crops and aligns all rasters
Crops each raster to the common bounds

Resamples them if needed to ensure exact pixel alignment (preserving NoData values)

## 4. Builds a 3D raster stack
Stack shape: (n_rasters, height, width)

Converts NoData to np.nan for safe statistics

## 5. Computes Per-Pixel Spatial MAE Across All Rasters
Calculates how much each pixel deviates from the average of all sessions

Outputs a color-coded MAE map per location (comparisons/MAE_{location}.png)

## 6. Computes Per-Pixel Standard Deviation
Computes np.nanstd(...) across the stack

Saves a histogram of the per-pixel STD values (comparisons/STD_threshold_{location}.png)

## 7. Identifies "unreliable" regions
Thresholds the STD map (default: 0.5 m)

Marks pixels with high STD as unreliable

Visualizes and saves a binary mask of those regions:

PNG: comparisons/unrealiable_areas_{location}.png

GeoTIFF: comparisons/{location}_unreliable_mask.tif



