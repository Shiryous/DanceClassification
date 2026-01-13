from PIL import Image
import os
import math

def split_spectrogram_horizontal(input_path, output_path, tile_width=128):
    """
    Splits a spectrogram (assuming fixed height) into segments of a specified width,
    including any smaller segment at the end.
    """
    try:
        # 1. Load the Spectrogram
        img = Image.open(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    
    # 2. Get Dimensions
    W, H = img.size
    if H != 128:
        print(f"Warning: Image height is {H}, not the expected 128. Proceeding with vertical crop of full height.")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    num_segments = math.ceil(W / tile_width) # Total segments, including the partial one
    tile_count = 0
    
    # 3. Loop and Crop
    for i in range(num_segments):
        # Define the bounding box for the crop (left, upper, right, lower)
        left = i * tile_width
        upper = 0 # Fixed top boundary
        
        # Calculate the right boundary (cannot exceed the total width W)
        right = min(left + tile_width, W)
        lower = H # Fixed bottom boundary
        
        # Determine the segment's actual width for reporting
        current_width = right - left
        if current_width != 128:
            break
        # Crop the image
        tile = img.crop((left, upper, right, lower))
        
        # 4. Save Segments
        # Naming convention: segment_index_WxH.png
        filename = f"{output_path[:-4]}_segment_{i}.png"
        tile.save(filename)
        
        tile_count += 1

    print(f"\nSuccessfully created {tile_count} segments in: {os.path.dirname(output_path)}")

## Folder Structure
DATASET_FOLDER = 'data/BallroomData'
SPECTROGRAM_FOLDER_PATH = f'{DATASET_FOLDER}/spectrograms'
SPLIT_SPECTROGRAM_FOLDER_PATH = f'{DATASET_FOLDER}/split_spectrograms'
ALL_BALLROOM_SPECTROGRAMS = f'{DATASET_FOLDER}/allBallroomSpectrograms'

try:
    with open(ALL_BALLROOM_SPECTROGRAMS, 'r') as file:
        image_paths = [line.strip() for line in file]
except FileNotFoundError:
    print(f"Error: The file '{ALL_BALLROOM_SPECTROGRAMS}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

for image_path in image_paths:
    output_folder = image_path.replace(SPECTROGRAM_FOLDER_PATH, SPLIT_SPECTROGRAM_FOLDER_PATH)
    split_spectrogram_horizontal(image_path, output_folder)