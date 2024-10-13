import pydicom
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def normalize_array(array):
    """Normalize array to 0-255 range."""
    min_val, max_val = array.min(), array.max()
    return ((array - min_val) / (max_val - min_val) * 255).astype(np.uint8)

def save_slice(slice_data, output_path):
    """Save a single slice as an image."""
    Image.fromarray(normalize_array(slice_data)).save(output_path)

def plot_middle_slice(pixel_array, output_folder):
    """Plot the middle slice of the DICOM image."""
    middle_slice = pixel_array[len(pixel_array) // 2] if len(pixel_array.shape) == 3 else pixel_array
    plt.imsave(os.path.join(output_folder, 'middle_slice_plot.png'), middle_slice, cmap='gray', format='png')

def dicom_to_images(dicom_file_path, output_folder):
    dicom_data = pydicom.dcmread(dicom_file_path)
    pixel_array = dicom_data.pixel_array
    
    if len(pixel_array.shape) == 2:
        pixel_array = np.expand_dims(pixel_array, axis=0)
    
    num_slices = pixel_array.shape[0]
    
    def process_slice(i):
        output_image_path = f"{output_folder}/slice_{i:03d}.png"
        save_slice(pixel_array[i], output_image_path)
    
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_slice, range(num_slices)), total=num_slices))
    
    plot_middle_slice(pixel_array, output_folder)
    print(f"Saved {num_slices} slices to {output_folder}")

# Example usage
dicom_file_path = 'TEST_SPECTRO_PHANTOM.MR.AMRI_LAB_OUD.0008.0002.2023.01.24.19.24.17.358231.98866525.IMA'
output_folder = 'output_images'
dicom_to_images(dicom_file_path, output_folder)