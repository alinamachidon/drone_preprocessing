import os
import numpy as np
import rasterio

# Water detection parameters
BLUE_THRESHOLD = 50  # Blue channel intensity threshold for water detection
WATER_PERCENTAGE = 10  # Percentage of water pixels to detect

def contains_water(image_path, blue_threshold=BLUE_THRESHOLD, water_percentage=WATER_PERCENTAGE):
    """Check if an image contains water based on blue channel intensity."""
    try:
        with rasterio.open(image_path) as src:
            img = src.read()  # Read as (bands, height, width)

        if img.shape[0] < 3:
            print(f"Skipping {image_path} (not RGB).")
            return False

        # Extract RGB channels
        red, green, blue = img[0], img[1], img[2]

        # Detect water pixels (high blue intensity)
        water_mask = (blue > blue_threshold) & (blue > red) 
        water_ratio = np.sum(water_mask) / water_mask.size * 100  # Percentage of water pixels

        return water_ratio > water_percentage  # Return True if water is dominant

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def is_mostly_white(image_path, threshold=0.9):
    """
    Checks if an image is completely or mostly (>90%) white.
    
    :param image_path: Path to the raster image file
    :param threshold: Percentage of white pixels required to classify as "mostly white"
    :return: True if mostly white, False otherwise
    """
    try:
        with rasterio.open(image_path) as src:
            img = src.read()  # Read as (bands, height, width)
            
        if img.shape[0] < 3:
            print(f"Skipping {image_path} (not RGB).")
            return False

        # Extract RGB channels
        red, green, blue = img[0], img[1], img[2]

        # Define white as pixels where all RGB values are close to 255
        white_pixels = (red >= 250) & (green >= 250) & (blue >= 250)
        white_ratio = np.sum(white_pixels) / white_pixels.size
        
        print(f"White pixel ratio: {white_ratio:.2%}")
        return white_ratio >= threshold
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return False

def extract_valid_tiles(input_folder):
    """Process all .tif files in the folder and detect water or white tiles."""
    # List all .tif files in the input folder
    tif_files = [f for f in os.listdir(input_folder) if f.endswith(".tif")]

    print(f"Processing {len(tif_files)} files in '{input_folder}'...")

    water_tiles = []

    for tif_file in tif_files:
        tif_path = os.path.join(input_folder, tif_file)
        
        # Check if the image is mostly white
        if is_mostly_white(tif_path):
            print(f"The image {tif_path} is mostly white.")
        else: 
            # Check if the image contains water
            if contains_water(tif_path):
                print(f"{tif_file} contains water!")
                water_tiles.append(tif_file)
            else:
                print(f"{tif_file} has no water.")

    return water_tiles

def save_results(water_tiles, output_file):
    """Save the list of water tiles to a text file."""
    try:
        with open(output_file, "w") as f:
            f.write("\n".join(water_tiles))
        print(f"\nSaved water tile list to '{output_file}'")
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    tile_size = 1440
    # Folder containing .tif files
    input_folder = f"data/{tile_size}/tiles"

    print(f"Input folder: {input_folder}")

    # Process images and detect water tiles
    water_tiles = extract_valid_tiles(input_folder)

    # Save the results to a file
    save_results(water_tiles, output_file=f"water_tiles_list_{tile_size}.txt")

if __name__ == "__main__":
    main()
