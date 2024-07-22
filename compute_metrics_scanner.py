# Computer PSNR for Dose
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
from utils import read_dicom

# ITK-Snap
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "16"

data_dir = '/mnt/nas4/datasets/ToCurate/QA4IQI/FinalDataset-TCIA-MultiCentric/Upl'
scanners = ['A1', 'A2', 'B1', 'B2', 'C1', 'D1', 'E1', 'E2', 'F1', 'G1', 'G2', 'H1', 'H2']
doses = ['1mGy', '3mGy', '6mGy', '10mGy', '14mGy']
dose = '10mGy'#, '14mGy']
reconstruction_method = ['*']#'FBP']#, 'IR', 'DL']
save_dir = './figures_scanners'
registration_mode = 'elastic'#'ants'#'elastic'# 'elastic'#'elstic' #None

scanners_list = ['A1', 'A2', 'B1', 'B2', 'C1', 'D1', 'E1', 'E2', 'F1', 'G1', 'G2', 'H1', 'H2']
thickness =     [2.0,  2.0,  2.0,   2.0, 2.0,  2.5,  2.0,  2.5,  2.5,  2.0,  2.0,  2.0,  2.0]
slice_thinknesses = {scanners_list[i]: thickness[i] for i in range(len(scanners_list))}

print('Registaration method: ', registration_mode)

level_window_torch = lambda x, level, window: torch.clamp(x,  level - window / 2, level + window/2)
#level_window = lambda x, level, window: np.clip((x - level + window / 2) / window, 0, 1)
level_window = lambda x, level, window: np.clip(x, level-window/2, level+window/2)
#level, window = 0, 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def flip_volume(volume):
    volume_flipped = np.zeros_like(volume)
    nslides = volume.shape[0]
    for i in range(nslides):
         volume_flipped[i] = volume[nslides - i - 1]
    return volume_flipped

# def read_dicom(dicom_dir, numpy_format=False):
#     # List to hold the image arrays
#     slices = []
#     filenames = sorted(os.listdir(dicom_dir))
#     # Iterate through all files in the directory
#     slice_locations = []
#     for filename in filenames:
#         if not 'mask' in filename and not '.json' in filename:
#             filepath = os.path.join(dicom_dir, filename)
#             # Read the DICOM file
#             ds = pydicom.dcmread(filepath)
#             # Extract the pixel array and add to the list
#             slices.append(ds)
#             slice_locations.append(float(ds.SliceLocation))
   
#     # Sort slices by Slice Location
#     slices.sort(key=lambda x: float(x.SliceLocation))
    
#     # Extract pixel data and stack into a 3D array
#     image_stack = np.stack([s.pixel_array for s in slices], axis=0)
#     image_stack = float(ds.RescaleSlope) * image_stack + float(ds.RescaleIntercept)
#     # Sort slices by Image Position Patient (z-axis position)
#     # slices.sort(key=lambda ds: ds.ImagePositionPatient[2])

#     # Sort slices by Slice Location
#     # sorted_indices = sorted(range(len(slice_locations)), key=lambda k: slice_locations[k])
#     # slices_reoreded = [slices[i] for i in sorted_indices]
#     #slices = [x for _, x in sorted(zip(slice_locations, slices))]
#     # if ['D1', 'E1'] in dicom_dir:
#     #     volume = image_stack
#     # else:
#     #     volume = flip_volume(image_stack)
#     volume = image_stack
#     if volume.shape[0] < 343:
#         # Calculate the zoom factors for each dimension
#         zoom_factors = (343 / volume.shape[0], 1, 1)
#         # Resample the image
#         volume = zoom(volume, zoom_factors, order=3) 
#     # Crop the region of Phantom
#     volume = volume[40:280, 120:395, 64:445]
#     # idx=-76;plt.imshow(np.clip(volume, -500, 1000)[idx,...], 'gray');plt.savefig('fig.png');
#     # Create a 3D numpy array from the sorted slices
#     # volume = np.stack([ds.pixel_array for ds in slices], axis=0)
#     #volume = float(ds.RescaleSlope) * volume + float(ds.RescaleIntercept)
#     #print(ds.RescaleSlope, ds.RescaleIntercept)
#     # if any([item in dicom_dir for item in ['D1', 'E2', 'F1', 'H2']]):
#     #     # volumes2 = np.zeros_like(volume)
#     #     # nslides = volume.shape[0]
#     #     # for i in range(nslides):
#     #     #     volumes2[i] = volume[nslides - i - 1]
#     #     # volume = volumes2
#     #     volume = np.flip(volume, axis=0)
#     volume_flipped = flip_volume(volume)
#     volume = level_window(volume, 500, 3000)
#     if numpy_format:
#         return [torch.tensor(volume.transpose(1, 2, 0)).to(device).float().unsqueeze(0).unsqueeze(0), volume, 
#             torch.tensor(volume_flipped.transpose(1, 2, 0)).to(device).float().unsqueeze(0).unsqueeze(0), volume_flipped]
#     else:
#         # Now 'volume' contains the 3D volume of the DICOM slices
#         # print(volume.shape)
#         return torch.tensor(volume.transpose(1, 2, 0)).to(device).float().unsqueeze(0).unsqueeze(0)

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
    img2, _, _  = find_shifts(img2, img1, axis=-1, downsample=16)
    return ssim(img1, img2)

# Main Fucntion
def main():
    psnr = metrics.PSNRMetric(max_val=2000)
    ssim = metrics.SSIMMetric(spatial_dims=3, data_range=2000)
    rmse = metrics.RMSEMetric()

    if not registration_mode is None and registration_mode.lower() == 'elastic':    
        # Define the parameters of the elastix registration:
        parameter_object = itk.ParameterObject.New()
        default_rigid_parameter_map = parameter_object.GetDefaultParameterMap('rigid')
        default_rigid_parameter_map['MaximumNumberOfIterations'] = ['1000']
        parameter_object.AddParameterMap(default_rigid_parameter_map)
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    zeros_matrix = np.zeros((len(scanners), len(scanners)))
    psnrs_mean_matrix, ssims_mean_matrix, rmses_mean_matrix = zeros_matrix.copy(), zeros_matrix.copy(), zeros_matrix.copy()
    psnrs_std_matrix, ssims_std_matrix, rmses_std_matrix = zeros_matrix.copy(), zeros_matrix.copy(), zeros_matrix.copy() 
    for i in range(len(scanners)):
        for j in range(len(scanners)):
            if i > j:
                continue
            print(scanners[i], 'compared with',scanners[j])
            files_scanner1 = sorted(glob(os.path.join(data_dir, scanners[i], f'*{dose}*{reconstruction_method[0]}*')))
            files_scanner2 = sorted(glob(os.path.join(data_dir, scanners[j], f'*{dose}*{reconstruction_method[0]}*')))
            files_scanner1 = [item for item in files_scanner1 if not 'mask' in item]
            files_scanner2 = [item for item in files_scanner2 if not 'mask' in item]
            if len(files_scanner1) != len(files_scanner2):
                files_scanner1 = files_scanner1[:min([len(files_scanner1), len(files_scanner2)])]
                files_scanner2 = files_scanner2[:min([len(files_scanner1), len(files_scanner2)])]
            
            psnrs, ssims, rmses = [], [], []
            #for file1, file2 in zip(files_scanner1, files_scanner2):
            for k in range(5):
                file1 = files_scanner1[k]
                file2 = files_scanner2[k]
                
                if registration_mode is None:
                    img1 = read_dicom(file1, slice_thinknesses=slice_thinknesses)
                    img2 = read_dicom(file2, slice_thinknesses=slice_thinknesses) 
                    if img1.shape != img2.shape:
                        print(f'Error: {file1} and {file2} have different shapes')
                        img1 = img1[:, :, :, :, :min([img1.shape[-1], img2.shape[-1]])]
                        img2 = img2[:, :, :, :, :min([img1.shape[-1], img2.shape[-1]])]
                elif registration_mode.lower() == 'elastic':
                    img1 = read_dicom(file1, numpy_format=True, slice_thinknesses=slice_thinknesses)
                    img2 = read_dicom(file2, numpy_format=True, slice_thinknesses=slice_thinknesses)
                    _ssim0 = rolled_ssim(img1[0], img2[0]).item()
                    _ssim1 = rolled_ssim(img1[0], img2[2]).item()
                    if _ssim1 > _ssim0:
                        img2[0] = img2[2]
                        img2[1] = img2[3]
                        print('Axial dimension flipped.')
                    _ssim = min(_ssim0, _ssim1)
                    #_ssim = ssim(level_window_torch(img1[0], -500, 1000), level_window_torch(img2[0], -500, 1000)).item()
                    print(f'SSIM Before Rolling {_ssim0:0.4f} ({_ssim1:0.4f})')
                    img2[0], shift, _ = find_shifts(img2[0], img1[0], axis=-1)
                    img2[1] = np.roll(img2[1], shift, axis=0)
                    _ssim = ssim(img1[0], img2[0]).item()
                    print(f'SSIM Before Registration {_ssim:0.4f}')
                    # plt.hist(img2[1].flatten());plt.hist(img1[1].flatten());plt.savefig('fig.png');plt.close()
                    # idx=172;plt.imshow(np.clip(img1[1], -500, 1000)[idx,...], 'gray');plt.savefig('fig.png');plt.imshow(np.clip(img2[1], -500, 1000)[idx,...], 'gray');plt.savefig('fig2.png');

                    img1, img2, _ = elstic_registration(img1[1], img2[1], parameter_object)
                    img1, img2 = elastic_results_to_tensor(img1, img2)

                elif registration_mode.lower() == 'ants':
                    img1 = ants.from_numpy(read_dicom(file1, numpy_format=True, slice_thinknesses=slice_thinknesses))
                    img2 = ants.from_numpy(read_dicom(file2, numpy_format=True, slice_thinknesses=slice_thinknesses))
                    registration_outs = ants.registration(fixed=img1, moving=img2, type_of_transform = 'SyN' )
                    img1, img2 = registration_outs['warpedfixout'], registration_outs['warpedmovout']
                    img1 = torch.tensor(img1.numpy()).to(device).float().unsqueeze(0).unsqueeze(0)
                    img2 = torch.tensor(img2.numpy()).to(device).float().unsqueeze(0).unsqueeze(0)
                else:
                    raise('Error: Registration mode not supported')
                    break
                
                _psnr = psnr(img1, img2).item()
                _ssim = ssim(img1, img2).item()
                _rmse = rmse(img1, img2).item()
                #if _ssim > 0.8:
                psnrs.append(_psnr)
                ssims.append(_ssim)
                rmses.append(_rmse)
                print(f'{_ssim:0.4f}')
            if len(psnrs) == 0:
                psnrs.append([-1, -1, -1])
                ssims.append([-1, -1, -1])
                rmses.append([-1, -1, -1])
            # Metri[i,j] = Metrci[j,i], However it is easier to debug this way:
            psnrs_mean_matrix[j, i] = np.mean(np.array(psnrs))
            ssims_mean_matrix[j, i] = np.mean(np.array(ssims))
            rmses_mean_matrix[j, i] = np.mean(np.array(rmses))
            psnrs_std_matrix[j, i] = np.std(np.array(psnrs))
            ssims_std_matrix[i, j] = np.std(np.array(ssims))
            rmses_std_matrix[i, j] = np.std(np.array(rmses))

    # Print the data for overleaf table
    for i in range(len(scanners)):
        line = [f'{psnrs_mean_matrix[i, j]:0.4f}+{psnrs_std_matrix[i, j]:0.4f} & ' for j in range(len(scanners)) if psnrs_mean_matrix[i, j] != 0 and j <= i]
        print(f'{scanners[i]} & ' + ''.join(line))
    for i in range(len(scanners)):
        line = [f'{ssims_mean_matrix[i, j]:0.6f}+{ssims_std_matrix[i, j]:0.6f} & ' for j in range(len(scanners)) if ssims_mean_matrix[i, j] != 0 and j <= i]
        print(f'{scanners[i]} & ' + ''.join(line))
    for i in range(len(scanners)):
        line = [f'{rmses_mean_matrix[i, j]:0.4f}+{rmses_std_matrix[i, j]:0.4f} & ' for j in range(len(scanners)) if rmses_mean_matrix[i, j] != 0 and j <= i]
        print(f'{scanners[i]} & ' + ''.join(line))
                
if __name__ == '__main__':
    main()