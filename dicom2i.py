import pydicom
from PIL import Image
import numpy as np

def dicom_to_images(dicom_file_path, output_folder):
    # Read the DICOM file
    dicom_data = pydicom.dcmread(dicom_file_path)

    # Get the pixel array from the DICOM data
    pixel_array = dicom_data.pixel_array

    # Check if the image is 3D
    if len(pixel_array.shape) == 3:
        num_slices = pixel_array.shape[0]
    else:
        num_slices = 1
        pixel_array = np.expand_dims(pixel_array, axis=0)

    for i in range(num_slices):
        # Get the slice
        slice_array = pixel_array[i]

        # Normalize the slice to 0-255 range
        normalized_array = ((slice_array - np.min(slice_array)) / 
                            (np.max(slice_array) - np.min(slice_array)) * 255)
        normalized_array = normalized_array.astype(np.uint8)

        # Create an image from the normalized array
        image = Image.fromarray(normalized_array)

        # Save the image as a PNG file
        output_image_path = f"{output_folder}/slice_{i:03d}.png"
        image.save(output_image_path)

    print(f"Saved {num_slices} slices to {output_folder}")

# Example usage
dicom_file_path = 'TEST_SPECTRO_PHANTOM.MR.AMRI_LAB_OUD.0008.0002.2023.01.24.19.24.17.358231.98866525.IMA'
output_folder = 'output_images'
dicom_to_images(dicom_file_path, output_folder)