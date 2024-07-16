# Create a registered dataset from a folder of images
# Usage: python3 create_registered_dataset.py <folder> <reference volume>

import os
import sys
import itk
import numpy as np
from glob import glob
from monai import metrics
from ImageRegistration import ImageRegistrator
from utils import read_dicom, save_nifti, rolled_ssim, find_shifts

# Default parameters for file selection
data_dir = '/mnt/nas4/datasets/ToCurate/QA4IQI/FinalDataset-TCIA-MultiCentric/Upl'
reference_volume_dir = ''
scanners = '*'#['A1', 'A2', 'B1', 'B2', 'C1', 'D1', 'E1', 'E2', 'F1', 'G1', 'G2', 'H1', 'H2']
dose = '10mGy'
reconstruction_method = '*'#'FBP'#, 'IR', 'DL'
dataset_dir = '/mnt/nas7/data/reza/registered_dataset'
registration_mode = 'elastic' #'ants'
ssim_data_range = 2000

def create_registered_dataset(folder, reference_volume):

    # Define the SSIM
    ssim = metrics.SSIMMetric(spatial_dims=3, data_range=ssim_data_range)

    # Read the reference volume
    reference_volumes = read_dicom(reference_volume_dir, numpy_format=True)

    # List of the images to be registered:
    scan_folder = sorted(glob(os.path.join(data_dir, scanners, f'*{dose}*{reconstruction_method}*')))

    # Create the registered dataset directory:
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    registrator = ImageRegistrator(registration_mode, reference_volumes[1])

    # Register all images in the folder:
    for folder in scan_folder:
        
        # Resulting image:
        final_file = os.path.join(dataset_dir, os.path.basename(folder) + '.nii.gz')
        if os.path.exists(final_file):
            print(f'{final_file} exists. Skipping...')
            continue
        
        # Read the image
        volumes = read_dicom(folder, numpy_format=True)
        
        # Check if the axial dimension is flipped
        _ssim0 = rolled_ssim(reference_volumes[0], volumes[0]).item()
        _ssim1 = rolled_ssim(reference_volumes[0], volumes[2]).item()
        if _ssim1 > _ssim0:
            volumes[0] = volumes[2]
            volumes[1] = volumes[3]
            print('Axial dimension flipped.')
        _ssim = min(_ssim0, _ssim1)
        print(f'SSIM Before Rolling {_ssim:0.4f}')
        
        # Find the shift between the images
        volumes[0], shift, _ = find_shifts(volumes[0], reference_volumes[0], axis=-1)
        volumes[1] = np.roll(img2[1], shift, axis=0)
        _ssim = ssim(reference_volumes[0], volumes[0]).item()
        print(f'SSIM Before Registration {_ssim:0.4f}')

        # Register the reference image
        registered_image = registrator.register_image(volumes[1])
        
        # Convert to tensor
        registered_image_tensor = registrator.to_tensor(registered_image)

        # Calculate the SSIM
        _ssim = ssim(reference_volumes[0], registered_image_tensor).item()
        print(f'{_ssim:0.4f}')
        
        # Save the registered image
        save_nifti(registered_image, os.path.join(dataset_dir, os.path.basename(folder) + '.nii.gz'))
    

if __name__ == '__main__':
    if not sys.argv[1] is None:
        folder = sys.argv[1]
    if not sys.argv[2] is None:
        reference_volume = sys.argv[2]
    create_registered_dataset(folder, reference_volume)