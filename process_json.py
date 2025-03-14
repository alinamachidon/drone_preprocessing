import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.warp import calculate_default_transform
import json
import pyproj
import rasterio
from rasterio.features import rasterize
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import geopandas as gpd

def get_crs_from_geojson(geojson_path):
    """
    Extracts the CRS from a GeoJSON file and converts it into an EPSG code or PROJ string.
    
    Args:
        geojson_path (str): Path to the GeoJSON file.
    
    Returns:
        str: EPSG code (e.g., 'EPSG:4326') or full PROJ string if EPSG is not available.
    """
    with open(geojson_path, "r") as f:
        geojson_data = json.load(f)

    crs_info = geojson_data.get("crs", {}).get("properties", {}).get("name", None)

    if crs_info:
        # Convert OGC CRS name to an EPSG code if possible
        if crs_info.startswith("urn:ogc:def:crs:OGC:1.3:CRS84"):
            return "EPSG:4326"  # CRS84 is equivalent to WGS84
        elif crs_info.startswith("urn:ogc:def:crs:EPSG::"):
            epsg_code = crs_info.split(":")[-1]
            return f"EPSG:{epsg_code}"
        else:
            # Try to get a PROJ string for unknown CRSs
            try:
                crs = pyproj.CRS(crs_info)
                return crs.to_proj4()
            except:
                return "Unknown CRS format"

    return "No CRS found in GeoJSON"


def rasterize_labels(geojson_path, input_raster, output_raster, target_resolution=10):
    """
    Rasterizes a GeoJSON file to match an input raster's CRS and resolution.
    
    Args:
        geojson_path (str): Path to the GeoJSON file containing labeled features.
        input_raster (str): Path to the raster file to match CRS and resolution.
        output_raster (str): Path to save the rasterized labels.
        target_resolution (float): Target resolution in meters.
    """
    # Load vector data
    gdf = gpd.read_file(geojson_path)

    with rasterio.open(input_raster) as src:
        original_crs = src.crs

        # Detect if the input CRS is EPSG:4326 and convert to a projected CRS
        if original_crs.to_string().startswith("EPSG:4326"):
            print("Detected EPSG:4326, converting to EPSG:3857 (Web Mercator)...")
            dst_crs = "EPSG:3857"
        else:
            dst_crs = original_crs

        # Calculate new transform and raster dimensions
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=target_resolution
        )

        # Update profile for rasterized labels
        profile = src.profile.copy()
        profile.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "count": 1,
            "dtype": np.uint8,
            "nodata": 0,
            "compress": "lzw"
        })

        # Reproject vector geometries if necessary
        if gdf.crs != dst_crs:
            gdf = gdf.to_crs(dst_crs)

        # Convert geometries to raster shapes
        shapes = [(geom, 1) for geom in gdf.geometry]

        # Rasterize the GeoJSON features
        rasterized_labels = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,  # Default label for no data
            all_touched=True,  # Assign pixels touched by geometry
            dtype=np.uint8
        )

        # Save the rasterized labels
        with rasterio.open(output_raster, "w", **profile) as dst:
            dst.write(rasterized_labels, 1)

        print(f"Rasterized labels saved to: {output_raster}")




# Load GeoJSON file
geojson_path = "river-wgs84.geojson"
gdf = gpd.read_file(geojson_path)


crs = get_crs_from_geojson(geojson_path)
print(f"Extracted CRS: {crs}")

# Define the label mapping
label_map = {"river": 1, "no-river": 0}

# Create shapes for rasterization
shapes = [(geom, label_map.get(feat['river'], 0)) for geom, feat in zip(gdf.geometry, gdf.to_dict(orient="records"))]

rasterize_labels(geojson_path, "rescaled_drone_1m.tif", "rasterized_labels.tif", target_resolution=1)


# # Rasterize the GeoJSON features
# rasterized_labels = rasterize(
#     shapes,
#     out_shape=raster_data.shape,
#     transform=src.transform,
#     fill=0,  # Default label for no data
#     all_touched=True,  # Assign pixels touched by geometry
#     dtype=np.uint8
# )

