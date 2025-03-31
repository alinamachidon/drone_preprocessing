import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import array_bounds
from rasterio.windows import from_bounds

# === Configuration ===
data_dir = ""  # Path to your raster folder
NODATA_VALUE = -32767
output_dir = "comparisons"
os.makedirs(output_dir, exist_ok=True)  # Ensure output folder exists
raster_files = glob.glob(os.path.join(data_dir, "*.tif"))

# === Utility Functions ===

def load_raster(path):
    """Load a raster, convert NoData to np.nan, and return the array and profile."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else NODATA_VALUE
        data = np.where(np.isclose(data, nodata, atol=1), np.nan, data)
        print(f"{path} min/max: {np.nanmin(data)}, {np.nanmax(data)}")
        return data, src.profile

def parse_filename(filename):
    """Parse date, location, and session info from raster filename."""
    base = os.path.basename(filename)
    match = re.match(r"(\d{8})_(\w+)_ESAprojekt(\d*)_DTM\.tif", base)
    if match:
        date = match.group(1)
        location = match.group(2)
        session = match.group(3) if match.group(3) else ""
        return date, location, f"ESAprojekt{session}"
    return None, None, None

def get_common_bounds(profiles):
    """Compute geographic intersection (common extent) of all rasters."""
    bounds_list = [array_bounds(p['height'], p['width'], p['transform']) for p in profiles]
    lefts, bottoms, rights, tops = zip(*bounds_list)
    common_bounds = (max(lefts), max(bottoms), min(rights), min(tops))
    if common_bounds[0] >= common_bounds[2] or common_bounds[1] >= common_bounds[3]:
        raise ValueError("No overlapping area between rasters.")
    return common_bounds

def crop_to_common_bounds(path, bounds):
    """Crop raster to specified bounds."""
    with rasterio.open(path) as src:
        window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        cropped_data = src.read(1, window=window)
        cropped_transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update({
            "height": cropped_data.shape[0],
            "width": cropped_data.shape[1],
            "transform": cropped_transform
        })
        return cropped_data, profile

def resample_with_mask(source_array, source_profile, target_profile, nodata_val=NODATA_VALUE):
    """Safely resample an array to match target raster shape and transform, preserving NoData."""
    destination = np.full((target_profile['height'], target_profile['width']), nodata_val, dtype=np.float32)
    source_array = np.where(source_array == nodata_val, np.nan, source_array)
    reproject(
        source=source_array,
        destination=destination,
        src_transform=source_profile['transform'],
        src_crs=source_profile['crs'],
        dst_transform=target_profile['transform'],
        dst_crs=target_profile['crs'],
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=nodata_val
    )
    return np.where(np.isclose(destination, nodata_val, atol=1), np.nan, destination)

def compute_spatial_mae_across_stack(stack, nodata_val=NODATA_VALUE):
    """Compute per-pixel MAE across stack, ignoring nodata and requiring ≥2 valid values."""
    masked_stack = np.where(np.isclose(stack, nodata_val, atol=10), np.nan, stack)
    valid_mask = np.sum(np.isfinite(masked_stack), axis=0) >= 2
    pixel_mean = np.full(masked_stack.shape[1:], np.nan, dtype=np.float32)
    pixel_mae = np.full(masked_stack.shape[1:], np.nan, dtype=np.float32)

    if np.any(valid_mask):
        valid_idx = np.where(valid_mask)
        pixel_mean[valid_idx] = np.nanmean(masked_stack[:, valid_idx[0], valid_idx[1]], axis=0)
        abs_diff = np.abs(masked_stack - pixel_mean)
        pixel_mae[valid_idx] = np.nanmean(abs_diff[:, valid_idx[0], valid_idx[1]], axis=0)

    return pixel_mae

# === Main Processing ===

def main():
    # Organize rasters by location and date
    raster_dict = defaultdict(dict)
    for f in raster_files:
        date, location, session = parse_filename(f)
        if location and date:
            raster_dict[location][date] = f

    for location, date_dict in raster_dict.items():
        print(f"\n--- {location} ---")
        session_keys = sorted(date_dict.keys())
        print(session_keys)

        # Load profiles to compute common bounds
        all_profiles = {k: load_raster(date_dict[k])[1] for k in session_keys}
        common_bounds = get_common_bounds(list(all_profiles.values()))

        # Use first raster as reference
        ref_key = session_keys[0]
        ref_data, ref_profile = crop_to_common_bounds(date_dict[ref_key], common_bounds)

        # Crop and align all rasters to the reference grid
        cropped_rasters = {}
        for k in session_keys:
            data, profile = crop_to_common_bounds(date_dict[k], common_bounds)
            if (profile['height'], profile['width']) != (ref_profile['height'], ref_profile['width']):
                print(f"Resampling {k} to match {ref_key}")
                data = resample_with_mask(data, profile, ref_profile)
            cropped_rasters[k] = data

        # Stack all cropped rasters (3D array)
        raster_stack = [r.astype(np.float32) for r in cropped_rasters.values()]
        stack = np.stack(raster_stack, axis=0)
        #stack_clean = np.where(np.isclose(stack, NODATA_VALUE, atol=10), np.nan, stack)
        stack_clean = np.where(stack == NODATA_VALUE, np.nan, stack)
        valid_mask = np.sum(np.isfinite(stack_clean), axis=0) >= 2

        # === Compute Per-Pixel MAE ===
        mae_map_all = compute_spatial_mae_across_stack(stack_clean)

        plt.figure(figsize=(8, 6))
        plt.imshow(mae_map_all, cmap="plasma", vmin=0, vmax=np.nanpercentile(mae_map_all, 99))
        plt.title(f"Per-pixel MAE across all rasters – {location}")
        plt.colorbar(label="MAE (m)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"MAE_{location}.png"))
        plt.close()

        # === Compute and Plot STD Histogram ===
        std_dev = np.nanstd(stack_clean, axis=0)
        std_values = std_dev[valid_mask].flatten()

        plt.figure(figsize=(8, 4))
        plt.hist(std_values, bins=100, color='steelblue', alpha=0.8)
        plt.title(f"STD Histogram – {location}")
        plt.xlabel("Standard Deviation of Depth (m)")
        plt.ylabel("Pixel Count")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"STD_threshold_{location}.png"))
        plt.close()

        # === Detect and Save Unreliable Areas ===
        threshold = 0.5  #could also be computed dynamically
        unreliable_mask = std_dev > threshold

        plt.figure(figsize=(8, 6))
        plt.imshow(unreliable_mask, cmap="gray")
        plt.title(f"Unreliable Depth Regions – {location}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"unreliable_areas_{location}.png"))
        plt.close()

        # Save unreliable mask as GeoTIFF
        mask_profile = ref_profile.copy()
        #mask_profile.update(dtype="uint8", count=1)
        with rasterio.open(os.path.join(output_dir, f"{location}_unreliable_mask.tif"), "w", **mask_profile) as dst:
            dst.write(unreliable_mask.astype("uint8"), 1)


if __name__ == "__main__":
    main()
