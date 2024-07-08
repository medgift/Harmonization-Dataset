# Computer PSNRs for Dose
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

# ITK-Snap

data_dir = '/mnt/nas4/datasets/ToCurate/QA4IQI/FinalDataset-TCIA-MultiCentric/Upl'
scanners = ['A1', 'A2', 'B1', 'B2', 'C1', 'D1', 'E1', 'E2', 'F1', 'G1', 'G2', 'H1', 'H2']
doses = ['1mGy', '3mGy', '6mGy', '10mGy', '14mGy']
dose = '10mGy'#, '14mGy']
reconstruction_method = ['FBP']#, 'IR', 'DL']
save_dir = './figures_scanners'
registration_mode = 'elastic'#'ants'# 'elastic'#'elstic' #None

print('Registaration method: ', registration_mode)

level_window_torch = lambda x, level, window: torch.clamp((x - level + window / 2) / window, 0, 1)
level_window = lambda x, level, window: np.clip((x - level + window / 2) / window, 0, 1)
level, window = 0, 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_dicom(dicom_dir, numpy_format=False):
    # List to hold the image arrays
    slices = []

    # Iterate through all files in the directory
    for filename in sorted(os.listdir(dicom_dir)):
        if not 'mask' in filename and not '.json' in filename:
            filepath = os.path.join(dicom_dir, filename)
            # Read the DICOM file
            ds = pydicom.dcmread(filepath)
            # Extract the pixel array and add to the list
            slices.append(ds)

    # Sort slices by Image Position Patient (z-axis position)
    # slices.sort(key=lambda ds: ds.ImagePositionPatient[2])

    # Create a 3D numpy array from the sorted slices
    volume = np.stack([ds.pixel_array for ds in slices], axis=0)
    volume = float(ds.RescaleSlope) * volume + float(ds.RescaleIntercept)
    #print(ds.RescaleSlope, ds.RescaleIntercept)
    if numpy_format:
        return volume
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

def elstic_registration(fixed_image_file, moving_image_file, parameter_object):
    fixed_image = itk.imread(fixed_image_file, itk.F)
    moving_image = itk.imread(moving_image_file, itk.F)
    # Load Elastix Image Filter Object
    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
    elastix_object.SetParameterObject(parameter_object)
    # Set additional options
    elastix_object.SetLogToConsole(False)
    # Update filter object (required)
    elastix_object.UpdateLargestPossibleRegion()
    # Results of Registration
    result_image = elastix_object.GetOutput()
    result_transform_parameters = elastix_object.GetTransformParameterObject()
    return fixed_image, result_image, result_transform_parameters

def elastic_results_to_tensor(fixed_image, moving_image):
    fixed_image = itk.GetArrayFromImage(fixed_image).transpose([1, 2, 0])
    moving_image = itk.GetArrayFromImage(moving_image).transpose([1, 2, 0])
    return torch.tensor(fixed_image).to(device).float().unsqueeze(0).unsqueeze(0), torch.tensor(moving_image).to(device).float().unsqueeze(0).unsqueeze(0)

def find_shifts(img, gt, axis=-1):
    # Shift the image on the last axis and compute rmses
    rmse = metrics.RMSEMetric()
    rmses = []
    for i in range(img.shape[axis]):
        img_shifted = torch.roll(img, shifts=i, dims=axis)
        rmses.append(rmse(img_shifted, gt).item())
    rmses = np.array(rmses)
    shift = np.argmin(rmses)
    return torch.roll(img, shifts=shift, dims=axis), shift, rmses

# Main Fucntion
def main():
    psnr = metrics.PSNRMetric(max_val=2000)
    ssim = metrics.SSIMMetric(spatial_dims=3, data_range=2000)
    rmse = metrics.RMSEMetric()

    if not registration_mode is None and registration_mode.lower() == 'elastic':
        # Define the parameters of the elastix registration:
        parameter_object = itk.ParameterObject.New()
        default_rigid_parameter_map = parameter_object.GetDefaultParameterMap('rigid')
        default_rigid_parameter_map['MaximumNumberOfIterations'] = ['10']
        parameter_object.AddParameterMap(default_rigid_parameter_map)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    zeros_matrix = np.zeros((len(scanners), len(scanners)))
    psrns_mean_matrix, ssims_mean_matrix, rmses_mean_matrix = zeros_matrix.copy(), zeros_matrix.copy(), zeros_matrix.copy()
    psrns_std_matrix, ssims_std_matrix, rmses_std_matrix = zeros_matrix.copy(), zeros_matrix.copy(), zeros_matrix.copy() 
    for i in range(len(scanners)):
        print(scanners[i])
        for j in range(len(scanners)):
            if i < j:
                continue
            files_scanner1 = sorted(glob(os.path.join(data_dir, scanners[i], f'*{dose}*{reconstruction_method}*')))
            files_scanner2 = sorted(glob(os.path.join(data_dir, scanners[j], f'*{dose}*{reconstruction_method}*')))
            if len(files_scanner1) != len(files_scanner2):
                files_scanner1 = files_scanner1[:min([len(files_scanner1), len(files_scanner2)])]
                files_scanner2 = files_scanner2[:min([len(files_scanner1), len(files_scanner2)])]
            
            psrns, ssims, rmses = [], [], []
            #for file1, file2 in zip(files_scanner1, files_scanner2):
            for k in range(3):
                file1 = files_scanner1[k]
                file2 = files_scanner2[k]
                
                if registration_mode is None:
                    img1 = read_dicom(file1)
                    img2 = read_dicom(file2)
                    if img1.shape != img2.shape:
                        print(f'Error: {file1} and {file2} have different shapes')
                        img1 = img1[:, :, :, :, :min([img1.shape[-1], img2.shape[-1]])]
                        img2 = img2[:, :, :, :, :min([img1.shape[-1], img2.shape[-1]])]
                elif registration_mode.lower() == 'elastic':
                    try:
                        img1, img2, _ = elstic_registration(file1, file2, parameter_object)
                        img1, img2 = elastic_results_to_tensor(img1, img2)
                    except:
                        print(f'Error: {file1} and {file2} have different shapes')
                        continue
                elif registration_mode.lower() == 'ants':
                    img1 = ants.from_numpy(read_dicom(file1, numpy_format=True))
                    img2 = ants.from_numpy(read_dicom(file2, numpy_format=True))
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
                psrns.append(_psnr)
                ssims.append(_ssim)
                rmses.append(_rmse)
                print(_ssim)
            if len(psrns) == 0:
                psrns.append([-1, -1, -1])
                ssims.append([-1, -1, -1])
                rmses.append([-1, -1, -1])
            psrns_mean_matrix[i, j] = np.median(np.array(psrns))
            ssims_mean_matrix[i, j] = np.median(np.array(ssims))
            rmses_mean_matrix[i, j] = np.median(np.array(rmses))
            psrns_std_matrix[i, j] = np.std(np.array(psrns))
            ssims_std_matrix[i, j] = np.std(np.array(ssims))
            rmses_std_matrix[i, j] = np.std(np.array(rmses))

    # Print the data for overleaf table
    for i in range(len(scanners)):
        line = [f'{psrns_mean_matrix[i, j]:0.2f}+{psrns_std_matrix[i, j]:0.2f} & ' for j in range(len(scanners)) if psrns_mean_matrix[i, j] != 0 and j <= i]
        print(f'{scanners[i]} & ' + ''.join(line))
    for i in range(len(scanners)):
        line = [f'{ssims_mean_matrix[i, j]:0.4f}+{ssims_std_matrix[i, j]:0.4f} & ' for j in range(len(scanners)) if ssims_mean_matrix[i, j] != 0 and j <= i]
        print(f'{scanners[i]} & ' + ''.join(line))
    for i in range(len(scanners)):
        line = [f'{rmses_mean_matrix[i, j]:0.2f}+{rmses_std_matrix[i, j]:0.2f} & ' for j in range(len(scanners)) if rmses_mean_matrix[i, j] != 0 and j <= i]
        print(f'{scanners[i]} & ' + ''.join(line))
                
if __name__ == '__main__':
    main()