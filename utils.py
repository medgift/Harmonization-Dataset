import os
import itk
from ants import image_read, registration
import ants
import torch
import pydicom
import numpy as np
from glob import glob
from monai import metrics
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import nibabel as nib
import dcmstack

level_window_torch = lambda x, level, window: torch.clamp(x,  level - window / 2, level + window/2)
#level_window = lambda x, level, window: np.clip((x - level + window / 2) / window, 0, 1)
level_window = lambda x, level, window: np.clip(x, level-window/2, level+window/2)
#level, window = 0, 1000

def flip_volume(volume, axis=0):
    volume = np.swapaxes(volume, 0, axis)
    volume_flipped = np.zeros_like(volume)
    nslices = volume.shape[0]
    for i in range(nslices):
         volume_flipped[i, ...] = volume[nslices - i - 1, ...]
    volume_flipped = np.swapaxes(volume_flipped, axis, 0)
    return volume_flipped

def replace_missing_values(ds):
    for elem in ds:
        if elem.value is None:
            elem.value = 11111111
    return ds

def dicom_to_nifti(dicom_dataset):
    # Extract pixel data and stack into a 3D array
    image_stack = dcmstack.DicomStack()
    # try:
    for ds in dicom_dataset:
        try:
            image_stack.add_dcm(ds)
        except:
            ds = replace_missing_values(ds)
            image_stack.add_dcm(ds)
    image_nifti = image_stack.to_nifti()
    return image_nifti

def read_dicom(dicom_dir, numpy_format=False, crop_region=[40,280,120,395,64,445], device='cuda', slice_thinknesses=None):
    # List to hold the image arrays
    slices = []
    filenames = sorted(os.listdir(dicom_dir))
    # Iterate through all files in the directory
    for filename in filenames:
        if not 'mask' in filename and not '.json' in filename:
            filepath = os.path.join(dicom_dir, filename)
            # Read the DICOM file
            ds = pydicom.dcmread(filepath)
            # Extract the pixel array and add to the list
            slices.append(ds)
   
    # Sort slices by Slice Location
    try:
        dicom_dataset = sorted(slices, key=lambda ds: -ds.InstanceNumber)
        image_nifti = dicom_to_nifti(dicom_dataset)
    except:
        dicom_dataset = sorted(slices, key=lambda ds: ds.SliceLocation)
        image_nifti = dicom_to_nifti(dicom_dataset)

    # Sort slices by Image Position Patient (z-axis position)
    # slices.sort(key=lambda ds: ds.ImagePositionPatient[2])

    # Sort slices by Slice Location
    # sorted_indices = sorted(range(len(slice_locations)), key=lambda k: slice_locations[k])
    # slices_reoreded = [slices[i] for i in sorted_indices]
    #slices = [x for _, x in sorted(zip(slice_locations, slices))]
    # if ['D1', 'E1'] in dicom_dir:
    #     volume = image_stack
    # else:
    #     volume = flip_volume(image_stack)
    volume = np.stack([ds.pixel_array for ds in dicom_dataset], axis=0)
    volume = float(ds.RescaleSlope) * volume + float(ds.RescaleIntercept)
    # if volume.shape[0] < 343:
    #     # Calculate the zoom factors for each dimension
    #     zoom_factors = (343 / volume.shape[0], 1, 1)
    #     # Resample the image
    #     volume = zoom(volume, zoom_factors, order=1)
    # if slice_thinknesses is not None:
    #     scanners_list = slice_thinknesses.keys()
    #     scanner_volume = [item for item in scanners_list if item in dicom_dir][0]
    # if slice_thinknesses[scanner_volume] != 2.0:
    slice_thinkness = float(ds.SliceThickness)
    if slice_thinkness != 2.0:
        zoom_factors = (slice_thinkness / 2.0, 1, 1)
        # Resample the image
        volume = zoom(volume, zoom_factors, order=1)
        if volume.shape[0] > 343:
            slices_2 = volume.shape[0]
            shift = (slices_2 - 343) // 2
            volume = volume[shift:shift+343, ...]
        
    # Crop the region of Phantom
    if crop_region:
        volume = volume[crop_region[0]:crop_region[1], crop_region[2]:crop_region[3], crop_region[4]:crop_region[5]]
    # idx=-76;plt.imshow(np.clip(volume, -500, 1000)[idx,...], 'gray');plt.savefig('fig.png');
    # Create a 3D numpy array from the sorted slices
    # volume = np.stack([ds.pixel_array for ds in slices], axis=0)
    #volume = float(ds.RescaleSlope) * volume + float(ds.RescaleIntercept)
    #print(ds.RescaleSlope, ds.RescaleIntercept)
    # if any([item in dicom_dir for item in ['D1', 'E2', 'F1', 'H2']]):
    #     # volumes2 = np.zeros_like(volume)
    #     # nslides = volume.shape[0]
    #     # for i in range(nslides):
    #     #     volumes2[i] = volume[nslides - i - 1]
    #     # volume = volumes2
    #     volume = np.flip(volume, axis=0)
    volume_flipped = flip_volume(volume)
    volume = level_window(volume, 500, 3000)
    if numpy_format:
        return [torch.tensor(volume.transpose(1, 2, 0)).to(device).float().unsqueeze(0).unsqueeze(0), volume, 
            torch.tensor(volume_flipped.transpose(1, 2, 0)).to(device).float().unsqueeze(0).unsqueeze(0), volume_flipped, image_nifti]
    else:
        # Now 'volume' contains the 3D volume of the DICOM slices
        # print(volume.shape)
        return torch.tensor(volume.transpose(1, 2, 0)).to(device).float().unsqueeze(0).unsqueeze(0)

def plot_save(img, title='fig.png'):
    slices = img.shape[-1]
    slice = level_window_torch(img[0, 0, :, :, slices//2], 0, 500).squeeze().cpu().numpy()
    plt.imshow(slice, cmap='gray')
    plt.axis('off')
    plt.savefig(title, bbox_inches='tight', pad_inches=0)
    plt.close()

def elstic_registration(fixed_image, moving_image, parameter_object):
    # Read the images from files
    fixed_image = itk.GetImageFromArray(fixed_image)
    moving_image = itk.GetImageFromArray(moving_image)
    # Downsample the images by a factor of 4
    downsample_factor = 4
    fixed_image_down = itk.bin_shrink_image_filter(fixed_image, shrink_factors=[downsample_factor, downsample_factor, downsample_factor])
    moving_image_down = itk.bin_shrink_image_filter(moving_image, shrink_factors=[downsample_factor, downsample_factor, downsample_factor])
    # Load Elastix Image Filter Object
    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image_down, moving_image_down)
    elastix_object.SetParameterObject(parameter_object)
    # Set additional options
    elastix_object.SetLogToConsole(False)
    # Update filter object (required)
    elastix_object.UpdateLargestPossibleRegion()
    # Computer the Registration Transformation
    result_transform_parameters = elastix_object.GetTransformParameterObject()
    # Apply the transform to the moving image
    result_image = elastix_object.GetOutput()
    return fixed_image_down, result_image, result_transform_parameters

def elastic_results_to_tensor(fixed_image, moving_image):
    fixed_image = itk.GetArrayFromImage(fixed_image).transpose([1, 2, 0])
    moving_image = itk.GetArrayFromImage(moving_image).transpose([1, 2, 0])
    return torch.tensor(fixed_image).to(device).float().unsqueeze(0).unsqueeze(0), torch.tensor(moving_image).to(device).float().unsqueeze(0).unsqueeze(0)

def array_to_tensor(nparray, device='cuda'):
    return torch.tensor(nparray).to(device).float().unsqueeze(0).unsqueeze(0)

def find_shifts(img, gt, axis=-1, downsample=4):
    # Shift the image on the last axis and compute rmses
    rmse = metrics.RMSEMetric()
    rmses = []
    for i in range(img.shape[axis]//downsample):
        img_shifted = torch.roll(img, shifts=i*downsample, dims=axis)
        rmses.append(rmse(img_shifted, gt).item())
    rmses = np.array(rmses)
    shift = np.argmin(rmses)*downsample
    return torch.roll(img, shifts=shift, dims=axis), shift, rmses

def rolled_ssim(img1, img2):
    ssim = metrics.SSIMMetric(spatial_dims=3, data_range=2000)
    img2, _, _  = find_shifts(img2, img1, axis=-1, downsample=8)
    return ssim(img1, img2)
