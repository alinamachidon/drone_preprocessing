
# Overview
The preprocessing.py script processes Sentinel-2 and drone imagery to ensure spatial alignment, extract matching tiles, and generate visualization comparisons. The pipeline includes reprojection, resampling, clipping, tiling, and visualization to facilitate accurate analysis between satellite and drone imagery.

# Features

Resampling: Adjust drone resolution to 1m.

Resampling: Adjust Sentinel-2 resolution to 10m.

Reprojection: Aligns the drone raster to the Sentinel-2 CRS.

Clipping: Crops images to a common spatial extent (defined by the geographic area covered by the drone image).

Tiling: Divides images into uniform size tiles (defined by the "tile_size" parameter).

Visualization: Compares corresponding Sentinel-2 and drone tiles.

Reconstruction: Merges tiles back into a full image for verification.

# Dependencies

Ensure the following Python libraries are installed (check requirements.txt):

pip install rasterio shapely geopandas numpy matplotlib opencv-python glob2

# Usage

## 1. Prepare the input data

Place the Sentinel-2 and drone raster files in the working directory:

Sentinel-2 raster: 2024-07-31_Sentinel-2_L2A.tif (should be a L2A S2 image with all bands, in the expected order: B01, B02, B03,..., B12)

Drone raster: 20240731_Volarje_RX1_orthomosaic_2cmGSD.tif (original drone image)

## 2. Run the script
Execute the script by running:
python preprocessing.py

## 3. Outputs
The script generates the following outputs:

Resampled drone raster: resampled_drone_1m.tif
Resampled Sentinel-2 raster: resampled_sentinel_10m.tif

Reprojected Drone Raster: reprojected_drone.tif

Clipped Rasters: soca_sentinel_clipped.tif, soca_drone_clipped.tif

Tiled images in corresponding folders:

sentinel_tiles/

drone_tiles/

Comparison plots: output_plots/

Reconstructed images:

sentinel_reconstructed.png

drone_reconstructed.png

# Processing steps:


## Step 1: Resample drone image to 1m
The drone raster is resampled to a 10 resolution using bilinear interpolation.

## Step 2: Resample Sentinel-2 image to 10m
The Sentinel-2 raster is resampled to a 10m resolution using bilinear interpolation.

## Step 3: Reproject drone image
The drone image is reprojected to match the Sentinel-2 CRS.

## Step 4: Clip images to the same spatial extent
Both Sentinel-2 and drone rasters are clipped using the bounding box of the drone raster.

## Step 5: Generate tiles
Each clipped raster is divided into fixed size tiles to ensure alignment.

## Step 6: Generate comparisons
For each corresponding tile:
The Sentinel-2 image is resized to match the drone tile.
A side-by-side visualization is saved in output_plots/.

## Step 7: Reconstruct full images
Tiles are stitched back together to verify correctness.

# Functions Overview

rescale_image(input_raster, output_raster, target_resolution): Resamples Sentinel-2/drone raster to the desired target_resolution (e.g. 1m, 10m).

reproject_raster(input_raster, reference_raster, output_raster): Reprojects drone raster.

clip_raster_and_save(raster_path, output_tif_path, reference_raster): Clips raster using drone extent.

create_tiles(raster_path, output_folder, tile_size_meters): Generates tiles from rasters.

plot_all_tiles(drone_folder, sentinel_folder, output_folder): Saves side-by-side tile comparisons.

plot_reconstructed_image(tile_folder, tile_size_meters, pixel_size, output_path): Reconstructs and visualizes full images.

# Notes

Ensure input rasters are georeferenced correctly.

Modify tile_size if a different tile size is needed.

Default tile size is 300m x 300m, but can be changed in the script.



