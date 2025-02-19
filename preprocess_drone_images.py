import rioxarray as rxr
import uavgeo as ug
import numpy as np
import os
from PIL import Image

def make_tiles(file_name, size_x, size_y, data_dir):
    """
    Splits an orthomosaic image into smaller tiles of given size.
    
    :param file_name: Path to the input orthomosaic image
    :param size_x: Tile width in pixels
    :param size_y: Tile height in pixels
    :return: List of NumPy arrays representing image tiles
    """
    try:
        ortho = rxr.open_rasterio(file_name)
    except Exception as e:
        print(f"Error loading file {file_name}: {e}")
    
    input_dims = {"x": size_x, "y": size_y}
    s_x, s_y = ortho.shape[2], ortho.shape[1]  # Image dimensions

    if (ortho.rio.crs==None):
        ortho.rio.write_crs("EPSG:XXXX", inplace=True) 
    
    crs = ortho.rio.crs
    
    chip_gdf = ug.compute.create_chip_bounds_gdf(input_dims=input_dims, shape_x=s_x, shape_y=s_y, crs=crs)
    chip_gdf = ug.compute.imgref_to_crsref_boxes(raster=ortho, gdf=chip_gdf)
    chip_gdf = chip_gdf.set_geometry(chip_gdf["c_geom"])
    
    chip_gdf["geometry"].to_file(f"{data_dir}/chips1.geojson", driver="GeoJSON")
    chip_gdf["c_geom"].to_file(f"{data_dir}/chips2.geojson", driver="GeoJSON")
    
    geo_series = chip_gdf.geometry
    
    chip_list = [
        ortho.rio.clip_box(minx=row.bounds[0], miny=row.bounds[1], maxx=row.bounds[2], maxy=row.bounds[3])
        for _, row in geo_series.items()
    ]
    
    chip_list = [img.sel(band=[1, 2, 3]) for img in chip_list]
    np_imgs = [np.transpose(img.values, (1, 2, 0)) for img in chip_list]
    
    return np_imgs

def write_tiles_to_files(path, img_list):
    """
    Saves a list of image tiles to disk.
    
    :param path: Directory path to save images
    :param img_list: List of NumPy arrays representing images
    """
    
    
    for i, img in enumerate(img_list):
        filename = f"{i:06d}.tif"
        filepath = os.path.join(path, filename)
        im = Image.fromarray(img)
        print(f"Saving image to {filepath}")
        im.save(filepath)

def main():
    # # Alternatively process all .tif files in the input folder
    # input_folder = "drone_images"
    # print(f"Processing {len(tif_files)} files in '{input_folder}'...")
    # tif_files = [f for f in os.listdir(input_folder) if f.endswith(".tif")]
    # for tif_file in tif_files:
    #     input_file = os.path.join(input_folder, tif_file)

    input_file = "../magicbathy/MagicBathyNet/20240731_Volarje_RX1_orthomosaic_2cmGSD.tif"
    data_dir = "data"
    tile_size_x, tile_size_y = 1440, 1440
    output_dir = f"{data_dir}/chips_{tile_size_x}"
    
    os.makedirs(output_dir, exist_ok=True)

    print("Generating tiles...")
    np_imgs = make_tiles(file_name=input_file, size_x=tile_size_x, size_y=tile_size_y, data_dir=data_dir)
    
    if np_imgs:
        print("Exporting tiles...")
        write_tiles_to_files(path=output_dir, img_list=np_imgs)
    else:
        print("No tiles were generated.")

if __name__ == "__main__":
    main()
