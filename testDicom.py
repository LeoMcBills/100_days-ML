import pydicom

# Path to your .IMA file
file_path = 'TEST_SPECTRO_PHANTOM.MR.AMRI_LAB_OUD.0008.0002.2023.01.24.19.24.17.358231.98866525.IMA'

try:
    dicom_data = pydicom.dcmread(file_path)
    print("File is a valid DICOM file!")
except Exception as e:
    print(f"Error reading the file as DICOM: {e}")

import numpy as np
from skimage import io
from PIL import Image
import os

def extract_tif_to_jpeg_slices(tif_path, output_dir):
    # Load the .tif file
    volume = io.imread(tif_path)
        
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
        
    # Iterate through each slice in the volume
    for i in range(volume.shape[0]):
        # Extract the slice
        slice_2d = volume[i]
            
        # Normalize the slice to 0-255 range for JPEG
        slice_2d = ((slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min()) * 255).astype(np.uint8)
            
        # Convert to PIL Image
        img = Image.fromarray(slice_2d)
            
        # Save as JPEG
        output_path = os.path.join(output_dir, f'slice_{i:03d}.jpg')
        img.save(output_path, 'JPEG', quality=95)
        
    print(f"Extracted {volume.shape[0]} slices to {output_dir}")

# Example usage:
extract_tif_to_jpeg_slices('attention-mri.tif', 'output_slices_directory')
