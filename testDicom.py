import pydicom

# Path to your .IMA file
file_path = 'TEST_SPECTRO_PHANTOM.MR.AMRI_LAB_OUD.0008.0002.2023.01.24.19.24.17.358231.98866525.IMA'

try:
    dicom_data = pydicom.dcmread(file_path)
    print("File is a valid DICOM file!")
except Exception as e:
    print(f"Error reading the file as DICOM: {e}")
