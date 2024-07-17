# Create a registered dataset from a folder of images
# Usage: python3 create_registered_dataset.py <folder> <reference volume>

import os
import sys
import shutil
import numpy as np
import nibabel as nib
from glob import glob
from monai import metrics
from ImageRegistration import ImageRegistrator
from utils import read_dicom, rolled_ssim, find_shifts, array_to_tensor, flip_volume

# Default parameters for file selection
data_dir = '/mnt/nas4/datasets/ToCurate/QA4IQI/FinalDataset-TCIA-MultiCentric/Upl'
reference_volume_dir = '/mnt/nas4/datasets/ToCurate/QA4IQI/FinalDataset-TCIA-MultiCentric/Upl/A1/A1_174008_691000_SOMATOM_Definition_Edge_ID23_Harmonized_10mGy_IR_NrFiles_343'
scanners = '*'#['A1', 'A2', 'B1', 'B2', 'C1', 'D1', 'E1', 'E2', 'F1', 'G1', 'G2', 'H1', 'H2']
dose = '10mGy'
reconstruction_method = '*'#'FBP'#, 'IR', 'DL'
dataset_dir = '/mnt/nas7/data/reza/registered_dataset/'
registration_mode = 'elastic' #'ants'
ssim_data_range = 2000
crop_region = [20,330,120,395,64,445]
downsample_factor = 1

def create_registered_dataset(folder, reference_volume):

    # Define the SSIM
    ssim = metrics.SSIMMetric(spatial_dims=3, data_range=ssim_data_range)

    # Read the reference volume
    reference_volumes = read_dicom(reference_volume_dir, numpy_format=True, crop_region=crop_region)

    # List of the images to be registered:
    scan_folders = sorted(glob(os.path.join(data_dir, scanners, f'*{dose}*{reconstruction_method}*')))

    # Create the registered dataset directory:
    os.makedirs(dataset_dir, exist_ok=True)
    #os.makedirs(dataset_dir+'masks', exist_ok=True)

    registrator = ImageRegistrator(registration_mode, reference_volumes[1])

    # Print the ground truth directory
    print(f'Reference Volume: {reference_volume_dir}')

    # Register all images in the folder:
    for folder in scan_folders:
        
        print(f'Registering {folder} ...')
        # Resulting image:
        final_file = os.path.join(dataset_dir, os.path.basename(folder) + '.nii.gz')
        if os.path.exists(final_file):
            print(f'{final_file} exists. Skipping...')
            continue
        
        # Read the image
        volumes = read_dicom(folder, numpy_format=True, crop_region=crop_region)
        nifti_image = volumes[-1]
        
        # Check if the axial dimension is flipped
        _ssim0 = rolled_ssim(reference_volumes[0], volumes[0]).item()
        _ssim1 = rolled_ssim(reference_volumes[0], volumes[2]).item()
        if _ssim1 > _ssim0:
            volumes[0] = volumes[2]
            volumes[1] = volumes[3]
            print('Axial dimension flipped.')
        _ssim = max(_ssim0, _ssim1)
        print(f'SSIM Before Rolling {_ssim:0.4f} ({_ssim1:0.4f})')
        
        # Find the shift between the images
        volumes[0], shift, _ = find_shifts(volumes[0], reference_volumes[0], axis=-1)
        volumes[1] = np.roll(volumes[1], shift, axis=0)
        _ssim = ssim(reference_volumes[0], volumes[0]).item()
        print(f'SSIM Before Registration {_ssim:0.4f}')

        # Register the reference image
        registered_image = registrator.register_image(volumes[1], downsample_factor=downsample_factor)[0]     
        
        # Convert to tensor
        registered_image_tensor = array_to_tensor(registered_image.transpose(1,2,0))

        # Calculate the SSIM
        _ssim = ssim(reference_volumes[0], registered_image_tensor).item()
        print(f'{_ssim:0.4f}')

        # Crop the shape out of the image
        #registered_image = registered_image[100:,...] 
        registered_image_nifti = registered_image.transpose(2,1,0)  
        registered_image_nifti = flip_volume(registered_image_nifti, axis=1)
        #registered_image_nifti = flip_volume(registered_image_nifti, axis=1)
        
        # Save the registered image
        registered_nifti = nib.Nifti1Image(registered_image_nifti.astype(float), nifti_image.affine)
        registered_nifti.to_filename('result.nii.gz')

        # Move the saved nifti to the dataset folder:
        shutil.move('result.nii.gz', final_file)
    
if __name__ == '__main__':
    create_registered_dataset(data_dir, reference_volume_dir)