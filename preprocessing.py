import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
import numpy as np
from shapely.geometry import Polygon, mapping
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import glob
import cv2


sentinel_bands=[4, 3, 2]

def reproject_raster(input_raster, reference_raster, output_raster):
    """
    Reprojects a raster to match the CRS of a reference raster.
    """
    with rasterio.open(reference_raster) as ref_src:
        target_crs = ref_src.crs  

    with rasterio.open(input_raster) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )

        profile = src.profile.copy()
        profile.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height
        })

        with rasterio.open(output_raster, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest  
                )

    print(f"Reprojected raster saved to {output_raster}")



def rescale_drone_image(input_path, output_path, target_resolution=1):
    """Rescales the drone image from a high resolution (e.g., 2cm/pixel) to a lower resolution (e.g., 1m/pixel)."""
    
    with rasterio.open(input_path) as src:

        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:3857", src.width, src.height, *src.bounds
        ) # Web Mercator (Meters)
        #print(f"Estimated resolution in meters: {transform.a}")
        

        # Compute scaling factor 
        scale_factor = target_resolution / transform.a 
        
        # Compute new dimensions
        new_width = int(src.width / scale_factor)
        new_height = int(src.height / scale_factor)
        
        # print(f"Original resolution: {src.res[0]}m/pixel")
        # print(f"Target resolution: {target_resolution}m/pixel")
        # print(f"New image size: {new_width} x {new_height}")

        # Compute new transform
        new_transform = src.transform * Affine.scale(scale_factor, scale_factor)

        # Update profile for output
        profile = src.profile
        profile.update({
            "width": new_width,
            "height": new_height,
            "transform": new_transform,
            "driver": "GTiff"  # Ensure correct output format
        })

        # Resample and write output
        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):  # Loop through bands
                data = src.read(i, out_shape=(new_height, new_width), resampling=Resampling.bilinear)
                dst.write(data, i)

        print(f"Rescaled image saved to {output_path}")



def resample_raster(input_raster, output_raster, target_resolution, resampling_method=Resampling.bilinear):
    """
    Resamples raster to a target resolution (e.g., 10m/pixel).
    
    Parameters:
        input_raster (str): Path to the input raster.
        output_raster (str): Path to save the resampled output raster.
        target_resolution (float): Target resolution in meters (e.g., 10 for Sentinel-2 L2A).
        resampling_method (rasterio.enums.Resampling): Resampling method (default: bilinear).
    """
    with rasterio.open(input_raster) as src:
        original_crs = src.crs
        #print(f"Original CRS: {original_crs}")

        # Step 1: Detect if CRS is in degrees (EPSG:4326) and convert to meters
        if original_crs.to_string().startswith("EPSG:4326"):
            print("Detected EPSG:4326 (degrees), converting to a metric projection (UTM)...")
            
            # Convert to an appropriate UTM zone (automatic selection)
            dst_crs = "EPSG:3857"  # Web Mercator (Meters)
        else:
            dst_crs = original_crs  # Already in meters

        # Step 2: Calculate new transform and size
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=target_resolution
        )

        # Step 3: Update profile
        profile = src.profile.copy()
        profile.update({
            "crs": dst_crs,  # Update to projected CRS
            "transform": transform,
            "width": width,
            "height": height,
            "dtype": src.dtypes[0],  # Preserve dtype
            "nodata": src.nodata,  # Keep nodata values
            "compress": "lzw"  # Enable compression
        })

        # Step 4: Resample with the correct resolution
        with rasterio.open(output_raster, "w", **profile) as dst:
            for i in range(1, src.count + 1):  # Loop through bands
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling_method  
                )

        print(f"Resampled {input_raster} to {target_resolution}m/pixel and saved to {output_raster}")

def clip_raster_and_save(raster_path, output_tif_path, reference_raster):
    """
    Clips a raster using the bounding box of a reference raster (drone image).
    Ensures both images cover the same area.
    """
    with rasterio.open(reference_raster) as ref_src:
        bbox = ref_src.bounds  

    # Convert bounding box to a polygon
    coordinates = [
        (bbox.left, bbox.bottom),
        (bbox.left, bbox.top),
        (bbox.right, bbox.top),
        (bbox.right, bbox.bottom),
        (bbox.left, bbox.bottom)
    ]

    gdf = gpd.GeoDataFrame(geometry=[Polygon(coordinates)], crs=ref_src.crs)

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs  
        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)

        geojson_geom = [mapping(gdf.geometry[0])]

        # Clip raster
        clipped_raster, clipped_transform = mask(src, geojson_geom, crop=True)
        profile = src.profile
        profile.update({
            "height": clipped_raster.shape[1],
            "width": clipped_raster.shape[2],
            "transform": clipped_transform
        })

        # Save the clipped raster
        with rasterio.open(output_tif_path, "w", **profile) as dst:
            dst.write(clipped_raster)

    print(f"Clipped raster saved to {output_tif_path}")


# def pad_raster_to_tile_size(raster_path, padded_raster_path, tile_size_meters):
#     """
#     Pads the raster so its dimensions are fully divisible by the tile size.
#     Padding is added to the right and bottom to maintain geographic alignment.
#     """
#     with rasterio.open(raster_path) as src:
#         transform = src.transform
#         pixel_size_x, pixel_size_y = transform.a, -transform.e  # Get pixel size
#         tile_size_x = int(tile_size_meters / pixel_size_x)
#         tile_size_y = int(tile_size_meters / pixel_size_y)

#         # Compute the required padding
#         pad_x = (tile_size_x - (src.width % tile_size_x)) % tile_size_x
#         pad_y = (tile_size_y - (src.height % tile_size_y)) % tile_size_y

#         new_width = src.width + pad_x
#         new_height = src.height + pad_y

#         # Handle missing NoData value
#         nodata_value = src.nodata if src.nodata is not None else 0  

#         # Create new padded dataset
#         profile = src.profile.copy()
#         profile.update({
#             "width": new_width,
#             "height": new_height,
#             "nodata": nodata_value  # Ensure NoData is set
#         })

#         with rasterio.open(padded_raster_path, "w", **profile) as dst:
#             # Read original data
#             data = src.read()

#             # Create padded data array filled with NoData value
#             padded_data = np.full((src.count, new_height, new_width), nodata_value, dtype=src.dtypes[0])

#             # Copy original data into the top-left part of the new array
#             padded_data[:, :src.height, :src.width] = data

#             # Write padded raster
#             dst.write(padded_data)

#     return padded_raster_path


def pad_raster_to_tile_size(raster_path, padded_raster_path, tile_size_meters, pad_value=None):
    """
    Pads the raster so its dimensions are divisible by the tile size.
    Padding is added to the right and bottom. The padding value can be specified or inferred.
    """
    with rasterio.open(raster_path) as src:
        transform = src.transform
        pixel_size_x, pixel_size_y = transform.a, -transform.e
        tile_size_x = int(tile_size_meters / pixel_size_x)
        tile_size_y = int(tile_size_meters / pixel_size_y)

        pad_x = (tile_size_x - (src.width % tile_size_x)) % tile_size_x
        pad_y = (tile_size_y - (src.height % tile_size_y)) % tile_size_y

        new_width = src.width + pad_x
        new_height = src.height + pad_y

        # Read original data
        data = src.read()
        dtype = src.dtypes[0]

        print(raster_path," ",dtype)
        # Choose padding value
        if pad_value is None:
            if dtype == 'uint8':
                pad_value = 255
            elif dtype == 'uint16':
                pad_value = 10000
            elif 'float' in dtype:
                pad_value = 1.0
            else:
                pad_value = 0  # fallback

        # Create padded array
        padded_data = np.full((src.count, new_height, new_width), pad_value, dtype=dtype)
        padded_data[:, :src.height, :src.width] = data

        # Update profile
        profile = src.profile.copy()
        profile.update({
            "width": new_width,
            "height": new_height,
            "nodata": None,  # You can also set to pad_value if needed
            "dtype": dtype
        })

        with rasterio.open(padded_raster_path, "w", **profile) as dst:
            dst.write(padded_data)

    print("Bottom-right pixel values (should be padding):", padded_data[:, -1, -1])

    return padded_raster_path



def create_tiles(raster_path, output_folder, tile_size_meters):
    """
    Cuts a raster into tiles of a given geographic size.
    Ensures tiles cover the exact same spatial area.
    """
    os.makedirs(output_folder, exist_ok=True)

    padded_raster_path = raster_path.replace(".tif", "_padded.tif")
    padded_raster_path = pad_raster_to_tile_size(raster_path, padded_raster_path, tile_size_meters)

    with rasterio.open(padded_raster_path) as src:
        transform = src.transform
        pixel_size_x, pixel_size_y = transform.a, -transform.e  # Get pixel size
        tile_size_x = int(tile_size_meters / pixel_size_x)
        tile_size_y = int(tile_size_meters / pixel_size_y)

        num_tiles_x = src.width // tile_size_x
        num_tiles_y = src.height // tile_size_y

        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                left = transform.c + (i * tile_size_x * pixel_size_x)
                top = transform.f - (j * tile_size_y * pixel_size_y)
                right = left + (tile_size_x * pixel_size_x)
                bottom = top - (tile_size_y * pixel_size_y)

                window = rasterio.windows.from_bounds(left, bottom, right, top, transform)
                tile = src.read(window=window)

                profile = src.profile.copy()
                profile.update({
                    "width": window.width,
                    "height": window.height,
                    "transform": rasterio.windows.transform(window, transform)
                })

                output_tile_path = f"{output_folder}/tile_{i}_{j}.tif"
                with rasterio.open(output_tile_path, "w", **profile) as dst:
                    dst.write(tile)

                print(f"Saved tile: {output_tile_path}")

def normalize_image(image):
    """Normalize image to [0, 1] range for correct visualization."""
    image = image.astype(np.float32)
    min_val = np.percentile(image, 2)  # Clip out extreme values
    max_val = np.percentile(image, 98)
    # Avoid division by zero if max_val == min_val
    if max_val - min_val == 0:
        return np.zeros_like(image)  # or return image
    
    image = np.clip((image - min_val) / (max_val - min_val), 0, 1)
    return image

def upscale_image(image, target_size):
    """Upscale small Sentinel image to match drone tile size."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)  # Nearest-neighbor keeps original pixels



def plot_all_tiles_with_labels(drone_folder, sentinel_folder, labels_folder, output_folder):
    """
    Plots side-by-side comparisons for all matching Sentinel-2, Drone, and Label tiles.

    Args:
        drone_folder (str): Path to the folder containing drone tiles.
        sentinel_folder (str): Path to the folder containing sentinel tiles.
        labels_folder (str): Path to the folder containing label tiles.
        output_folder (str): Path to save the comparison PNGs.
    """
    os.makedirs(output_folder, exist_ok=True)

    drone_tiles = sorted(glob.glob(os.path.join(drone_folder, "*.tif")))
    sentinel_tiles = sorted(glob.glob(os.path.join(sentinel_folder, "*.tif")))
    label_tiles = sorted(glob.glob(os.path.join(labels_folder, "*.tif")))

    for drone_path, sentinel_path, label_path in zip(drone_tiles, sentinel_tiles, label_tiles):
        tile_name = os.path.basename(drone_path).replace(".tif", "")

        with rasterio.open(sentinel_path) as src:
            sentinel_data = src.read(sentinel_bands)
            sentinel_data = np.moveaxis(sentinel_data, 0, -1)  # Convert to (H, W, C)
            sentinel_data = normalize_image(sentinel_data)

        with rasterio.open(drone_path) as src:
            drone_data = src.read([1, 2, 3])  # Use first 3 bands as RGB
            drone_data = np.moveaxis(drone_data, 0, -1)  # Convert to (H, W, C)
            drone_data = normalize_image(drone_data)

        with rasterio.open(label_path) as src:
            labels_data = src.read(1)  # Assuming single-band label raster
            labels_data = normalize_image(labels_data)  # Normalize for visualization

        # Upscale Sentinel-2 Image and Labels to match Drone size
        upscaled_sentinel = upscale_image(sentinel_data, (drone_data.shape[1], drone_data.shape[0]))
        upscaled_labels = upscale_image(labels_data, (drone_data.shape[1], drone_data.shape[0]))

        # Plot side-by-side
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(upscaled_sentinel)
        axes[0].set_title(f"Sentinel-2: {tile_name}")
        axes[0].axis("off")

        axes[1].imshow(drone_data)
        axes[1].set_title(f"Drone: {tile_name}")
        axes[1].axis("off")

        axes[2].imshow(upscaled_labels, cmap="gray")
        axes[2].set_title(f"Labels: {tile_name}")
        axes[2].axis("off")

        plt.tight_layout()

        output_png_path = os.path.join(output_folder, f"{tile_name}.png")
        plt.savefig(output_png_path, dpi=300)
        plt.close(fig)
        
        print(f"Saved comparison: {output_png_path}")


def plot_all_tiles(drone_folder, sentinel_folder, output_folder):
    """
    Plots side-by-side comparisons for all matching Sentinel-2 and Drone tiles.

    Args:
        drone_folder (str): Path to the folder containing drone tiles.
        sentinel_folder (str): Path to the folder containing sentinel tiles.
        output_folder (str): Path to save the comparison PNGs.
    """
    os.makedirs(output_folder, exist_ok=True)

    drone_tiles = sorted(glob.glob(os.path.join(drone_folder, "*.tif")))
    sentinel_tiles = sorted(glob.glob(os.path.join(sentinel_folder, "*.tif")))

    for drone_path, sentinel_path in zip(drone_tiles, sentinel_tiles):
        tile_name = os.path.basename(drone_path).replace(".tif", "")

        with rasterio.open(sentinel_path) as src:
            sentinel_data = src.read(sentinel_bands)
            sentinel_data = np.moveaxis(sentinel_data, 0, -1)  # Convert to (H, W, C)
            sentinel_data = normalize_image(sentinel_data)

        # Load Drone Image (4-band RGBA)
        with rasterio.open(drone_path) as src:
            drone_data = src.read([1, 2, 3])  # Use first 3 bands as RGB
            drone_data = np.moveaxis(drone_data, 0, -1)  # Convert to (H, W, C)
            drone_data = normalize_image(drone_data)

        # Upscale Sentinel-2 Image to match Drone size
        upscaled_sentinel = upscale_image(sentinel_data, (drone_data.shape[1], drone_data.shape[0]))

        # Plot side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(upscaled_sentinel)
        axes[0].set_title(f"Sentinel-2: {tile_name}")
        axes[0].axis("off")

        axes[1].imshow(drone_data)
        axes[1].set_title(f"Drone: {tile_name}")
        axes[1].axis("off")

        plt.tight_layout()

        output_png_path = os.path.join(output_folder, f"{tile_name}.png")
        plt.savefig(output_png_path, dpi=300)
        plt.close(fig)
        
        print(f"Saved comparison: {output_png_path}")



def get_tile_bounds(image_path, col_off=0, row_off=0, width=None, height=None):
    """
    Extracts the geographic coordinates of a tile from a georeferenced image.
    """
    with rasterio.open(image_path) as src:
        width = width or src.width
        height = height or src.height
        window = Window(col_off, row_off, width, height)
        transform = src.window_transform(window)
        left, top = transform * (0, 0)
        right, bottom = transform * (width, height)
        return left, top, right, bottom  # (xmin, ymax, xmax, ymin)


def quantify_geographic_difference(drone_bounds, sentinel_bounds):
    """
    Computes the difference in geographic bounds between the drone and Sentinel tiles
    in centimeters.
    """
    # Convert differences from meters to centimeters
    diff_xmin = abs(drone_bounds[0] - sentinel_bounds[0]) * 100
    diff_ymax = abs(drone_bounds[1] - sentinel_bounds[1]) * 100
    diff_xmax = abs(drone_bounds[2] - sentinel_bounds[2]) * 100
    diff_ymin = abs(drone_bounds[3] - sentinel_bounds[3]) * 100

    return diff_xmin, diff_ymax, diff_xmax, diff_ymin

def plot_all_tiles_grid(drone_folder, sentinel_folder):
    """
    Plots a grid of all Sentinel-2 and Drone tiles, ensuring corresponding pairs are displayed together.
    Also checks if tiles cover the same geographic extent.
    """
    drone_tiles = sorted(glob.glob(os.path.join(drone_folder, "*.tif")))
    sentinel_tiles = sorted(glob.glob(os.path.join(sentinel_folder, "*.tif")))

    num_tiles = min(len(drone_tiles), len(sentinel_tiles))  # Ensure equal tile count
    fig, axes = plt.subplots(num_tiles, 2, figsize=(10, 5 * num_tiles))  # 2 columns: Sentinel & Drone
    if num_tiles == 1:
        axes = [axes]  

    for i, (drone_path, sentinel_path) in enumerate(zip(drone_tiles, sentinel_tiles)):
        tile_name = os.path.basename(drone_path).replace(".tif", "")
        
        # Validate geographic coverage
        #drone_bounds = get_tile_bounds(drone_path)
        #sentinel_bounds = get_tile_bounds(sentinel_path)

        #diff_xmin, diff_ymax, diff_xmax, diff_ymin = quantify_geographic_difference(drone_bounds, sentinel_bounds)

        # print(f"Tile {tile_name} geographic differences (cm):")
        # print(f"  xmin difference: {diff_xmin:.2f} cm")
        # print(f"  ymax difference: {diff_ymax:.2f} cm")
        # print(f"  xmax difference: {diff_xmax:.2f} cm")
        # print(f"  ymin difference: {diff_ymin:.2f} cm")

        # Load Sentinel-2 Image 
        with rasterio.open(sentinel_path) as src:
            sentinel_data = src.read(sentinel_bands)  # Assume RGB bands
            sentinel_data = np.moveaxis(sentinel_data, 0, -1)  # Convert to (H, W, C)
            sentinel_data = normalize_image(sentinel_data)

        # Load Drone Image 
        with rasterio.open(drone_path) as src:
            drone_data = src.read([1, 2, 3])  
            drone_data = np.moveaxis(drone_data, 0, -1)  # Convert to (H, W, C)
            drone_data = normalize_image(drone_data)

        # Upscale Sentinel-2 Image to match Drone size
        upscaled_sentinel = upscale_image(sentinel_data, (drone_data.shape[1], drone_data.shape[0]))

        axes[i][0].imshow(upscaled_sentinel)
        axes[i][0].set_title(f"Sentinel-2: {tile_name}")
        axes[i][0].axis("off")

        axes[i][1].imshow(drone_data)
        axes[i][1].set_title(f"Drone: {tile_name}")
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.savefig("all_tiles_plot.png")
    plt.close(fig)


def plot_reconstructed_image(tile_folder, tile_size_meters, pixel_size, output_path=None):
    """
    Reconstructs and plots all tiles of an image by stitching them back together.
    
    Args:
        tile_folder (str): Path to the folder containing tiles.
        tile_size_meters (int): The real-world size of each tile (e.g., 100m, 200m).
        pixel_size (float): Pixel size in meters (e.g., 10m for Sentinel, ~1m for Drone).
        output_path (str, optional): Path to save the reconstructed image.
    """
    tile_paths = sorted(glob.glob(os.path.join(tile_folder, "*.tif")))

    # Extract tile indices from filenames (assuming format: tile_X_Y.tif)
    tile_info = []
    for path in tile_paths:
        filename = os.path.basename(path)
        parts = filename.replace(".tif", "").split("_")
        try:
            x_idx, y_idx = int(parts[1]), int(parts[2])
            tile_info.append((x_idx, y_idx, path))
        except ValueError:
            print(f"Invalid file: {filename}")
    
    tile_info.sort(key=lambda x: (x[1], x[0]))

    num_tiles_x = max(x for x, _, _ in tile_info) + 1
    num_tiles_y = max(y for _, y, _ in tile_info) + 1

    #print(f"Detected Grid: {num_tiles_x} x {num_tiles_y} tiles")

    # Read the first tile to determine expected tile dimensions
    with rasterio.open(tile_info[0][2]) as src:
        expected_tile_height, expected_tile_width = src.shape

    # Create an empty array for the full reconstructed image (initially zeros)
    full_height = num_tiles_y * expected_tile_height
    full_width = num_tiles_x * expected_tile_width
    full_image = np.zeros((full_height, full_width), dtype=np.float32)

    for x_idx, y_idx, path in tile_info:
        with rasterio.open(path) as src:
            tile_data = src.read(1)  # Assuming single-band grayscale
            tile_height, tile_width = tile_data.shape  # Get actual tile size

        # Compute placement in the final array
        start_x, start_y = x_idx * expected_tile_width, y_idx * expected_tile_height

        # Adjust for tiles that are smaller than expected
        full_image[start_y:start_y + tile_height, start_x:start_x + tile_width] = tile_data[:tile_height, :tile_width]

    # Plot the reconstructed image
    plt.figure(figsize=(12, 8))
    plt.imshow(full_image, cmap='gray')
    plt.title(f"Reconstructed Image ({num_tiles_x}x{num_tiles_y} tiles)")
    plt.axis("off")

    # Save the reconstructed image 
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved reconstructed image to {output_path}")

def print_raster_info(raster_name):
    with rasterio.open(raster_name) as src:
        print(f"CRS: {src.crs}")
        print(f"Pixel Size: {src.res}")  
        print(f"Image Size: {src.width} x {src.height}")


def plot_drone_and_mask(drone_tile_path, mask_tile_path, output_path):
    """
    Plots the drone tile and its validity mask side by side.
    
    Args:
        drone_tile_path (str): Path to the drone image tile (GeoTIFF).
        mask_tile_path (str): Path to the corresponding validity mask (GeoTIFF).
        output_path (str, optional): If given, saves the plot to this path.
    """
    with rasterio.open(drone_tile_path) as drone_src:
        drone_data = drone_src.read([1, 2, 3])
        drone_data = np.moveaxis(drone_data, 0, -1)  # (H, W, C)
        drone_data = normalize_image(drone_data)

    with rasterio.open(mask_tile_path) as mask_src:
        mask_data = mask_src.read(1)  # single-band mask

    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(drone_data)
    axes[0].set_title("Drone Tile")
    axes[0].axis("off")

    axes[1].imshow(mask_data, cmap="gray")
    axes[1].set_title("Validity Mask (1 = valid, 0 = white)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved comparison plot to {output_path}")
  


def generate_validity_masks(drone_tiles_folder, output_mask_folder):
    """
    Generates validity masks for drone tiles by detecting pure white regions (R=G=B=255).
    
    Args:
        drone_tiles_folder (str): Path to folder with drone image tiles.
        output_mask_folder (str): Path to save binary validity masks.
    """
    os.makedirs(output_mask_folder, exist_ok=True)
    drone_tiles = sorted(glob.glob(os.path.join(drone_tiles_folder, "*.tif")))

    for tile_path in drone_tiles:
        tile_name = os.path.basename(tile_path)
        with rasterio.open(tile_path) as src:
            rgb = src.read([1, 2, 3])  # Shape: (3, H, W)
            profile = src.profile

        # Step 1: Identify strict white pixels (seed points)
        strict_white = np.all(rgb == 255, axis=0)  # shape: (H, W)

        # Step 2: Identify "almost white" pixels — candidates for invalid regions
        almost_white_threshold = 200
        almost_white = np.all(rgb >= almost_white_threshold, axis=0).astype(np.uint8)

        # Step 3: Label connected components in the almost-white mask
        num_labels, labels = cv2.connectedComponents(almost_white, connectivity=8)

        # Step 4: Create the final invalid mask
        final_invalid_mask = np.zeros_like(strict_white, dtype=np.uint8)

        for label in range(1, num_labels):  # Skip background label 0
            region = labels == label
            if np.any(strict_white[region]):  # If this region contains any strict white pixel
                final_invalid_mask[region] = 1  # Include the full region as invalid


        # Step 5: Detect black pixels (potential padding)
        black_mask = np.all(rgb == 0, axis=0)  # shape: (H, W)
        final_invalid_mask[black_mask] = 1

        # Step 6: Invert the mask to get the validity mask
        validity_mask = (1 - final_invalid_mask).astype(np.uint8)

        # Save as single-band raster
        profile.update({
            "count": 1,
            "dtype": "uint8"
        })

        mask_path = os.path.join(output_mask_folder, tile_name)
        with rasterio.open(mask_path, "w", **profile) as dst:
            dst.write(validity_mask[np.newaxis, :, :])  # Add band dim

        print(f"Saved validity mask: {mask_path}")

        drone_tile_path = os.path.join(drone_tiles_folder, tile_name)
        print(drone_tile_path)

        mask_vs_drone_path = os.path.join(output_mask_folder, f"mask_vs_drone_path_{tile_name}.png")
        plot_drone_and_mask(drone_tile_path, mask_path, mask_vs_drone_path)


def main():
     # Set this flag to 1 if label data is available, 0 otherwise
    labels_available = 1  # <-- change to 0 if labels are not available

    # Paths to input raster files
    sentinel_raster = "2024-07-31_Sentinel-2_L2A.tif"
    drone_raster = "../magicbathy/MagicBathyNet/20240731_Volarje_RX1_orthomosaic_2cmGSD.tif"

    # Target resolution for the drone image (in meters)
    target_drone_res = 1
    # Target resolution for the satellite image (in meters)
    target_s2_res = 10

    drone_rescaled = "rescaled_drone_1m.tif"
    resample_raster(drone_raster, drone_rescaled, target_drone_res)
    print_raster_info(drone_raster)
    
    sentinel_resampled = "rescaled_sentinel_10m.tif"
    resample_raster(sentinel_raster, sentinel_resampled, target_s2_res)
   
    sentinel_clipped = "soca_sentinel_clipped.tif"

    clip_raster_and_save(sentinel_resampled, sentinel_clipped, drone_rescaled)
    print_raster_info(sentinel_clipped)

    labels = "rasterized_labels.tif"
    print_raster_info(sentinel_clipped)

    labels_clipped = "labels_clipped.tif"
    clip_raster_and_save(labels, labels_clipped, drone_rescaled)
    print_raster_info(labels_clipped)

    sentinel_tiles_folder = "sentinel_tiles" # output folder for generated sentinel image tiles
    drone_tiles_folder = "drone_tiles" # output folder for generated drone image tiles
    labels_folder = "labels_tiles" # output folder for generated labels image tiles

    # target tile size (in meters)
    tile_size = 400
    create_tiles(sentinel_clipped, sentinel_tiles_folder, tile_size)
    create_tiles(drone_rescaled, drone_tiles_folder, tile_size)

    if labels_available:
        labels = "rasterized_labels.tif"
        labels_clipped = "labels_clipped.tif"

        print_raster_info(labels)
        clip_raster_and_save(labels, labels_clipped, drone_rescaled)
        print_raster_info(labels_clipped)

        create_tiles(labels_clipped, labels_folder, tile_size)
        plot_all_tiles_with_labels(drone_tiles_folder, sentinel_tiles_folder, labels_folder, "output_plots")
        
    else:
        print("Labels not available. Skipping label tile creation and plots.")

    validity_mask_folder = "validity_masks"
    generate_validity_masks(drone_tiles_folder, validity_mask_folder)

    plot_all_tiles(drone_tiles_folder, sentinel_tiles_folder, "output_plots")
    plot_all_tiles_grid(drone_tiles_folder, sentinel_tiles_folder)

    plot_reconstructed_image(tile_folder=sentinel_tiles_folder, tile_size_meters=tile_size, pixel_size=target_s2_res, output_path="sentinel_reconstructed.png")
    plot_reconstructed_image(tile_folder=drone_tiles_folder, tile_size_meters=tile_size, pixel_size=target_drone_res, output_path="drone_reconstructed.png")



if __name__ == "__main__":
    main()