# drone_preprocessing
Overview

The preprocessing.py script processes Sentinel-2 and drone imagery to ensure spatial alignment, extract matching tiles, and generate visualization comparisons. The pipeline includes reprojection, resampling, clipping, tiling, and visualization to facilitate accurate analysis between satellite and drone imagery.

Features:

Resampling: Adjust Sentinel-2 resolution to 10m.

Reprojection: Aligns the drone raster to the Sentinel-2 CRS.

Clipping: Crops images to a common spatial extent.

Tiling: Divides images into uniform 400m x 400m tiles.

Visualization: Compares corresponding Sentinel-2 and drone tiles.

Reconstruction: Merges tiles back into a full image for verification.

Dependencies

Ensure the following Python libraries are installed:

pip install rasterio shapely geopandas numpy matplotlib opencv-python glob2

Usage

1. Prepare Input Data

Place the Sentinel-2 and drone raster files in the working directory:

Sentinel-2 raster: 2024-07-31_Sentinel-2_L2A_(Raw)_stack.tif

Drone raster: rescaled_drone.tif (drone image, original 0.02m/pixel or a rescaled version 1m/pixel resolution for faster processing)

2. Run the Script

Execute the script by running:

python raster_processing.py

3. Outputs

The script generates the following outputs:

Resampled Sentinel-2 Raster: rescaled_sentinel_10m.tif

Reprojected Drone Raster: reprojected_drone.tif

Clipped Rasters: soca_sentinel_clipped.tif, soca_drone_clipped.tif

Tiled Images:

sentinel_tiles/

drone_tiles/

Comparison Plots: output_plots/

Reconstructed Images:

sentinel_reconstructed.png

drone_reconstructed.png

Processing Steps

Step 1: Resample Sentinel-2 to 10m

The Sentinel-2 raster is resampled to a 10m resolution using bilinear interpolation.

Step 2: Reproject Drone Image

The drone image is reprojected to match the Sentinel-2 CRS.

Step 3: Clip Images to the Same Spatial Extent

Both Sentinel-2 and drone rasters are clipped using the bounding box of the drone raster.

Step 4: Generate Tiles

Each clipped raster is divided into 400m x 400m tiles to ensure alignment.

Step 5: Generate Comparisons

For each corresponding tile:

The Sentinel-2 image is resized to match the drone tile.

A side-by-side visualization is saved in output_plots/.

Step 6: Reconstruct Full Images

Tiles are stitched back together to verify correctness.

Functions Overview

resample_sentinel_to_10m(input_raster, output_raster): Resamples Sentinel-2 to 10m.

reproject_raster(input_raster, reference_raster, output_raster): Reprojects drone raster.

clip_raster_and_save(raster_path, output_tif_path, reference_raster): Clips raster using drone extent.

create_tiles(raster_path, output_folder, tile_size_meters): Generates tiles from rasters.

plot_all_tiles(drone_folder, sentinel_folder, output_folder): Saves side-by-side tile comparisons.

plot_reconstructed_image(tile_folder, tile_size_meters, pixel_size, output_path): Reconstructs and visualizes full images.

Notes

Ensure input rasters are georeferenced correctly.

Modify tile_size_meters if a different tile size is needed.

Default tile size is 400m x 400m, but can be changed in the script.

Author

Developed for geospatial image analysis and remote sensing applications.
