# Computer PSNRs for Dose
import os
import torch
import pydicom
import numpy as np
from glob import glob
from monai import metrics
import matplotlib.pyplot as plt

# ITK-Snap

data_dir = '/mnt/nas4/datasets/ToCurate/QA4IQI/FinalDataset-TCIA-MultiCentric/Upl'
scanners = ['H2']#['A1', 'A2', 'B1', 'B2', 'C1', 'D1', 'E1', 'E2', 'F1', 'G1', 'G2', 'H1', 'H2']
doses = ['1mGy', '3mGy', '6mGy', '10mGy', '14mGy']
doses = ['10mGy', '14mGy']
reconstruction_method = ['FBP']#, 'IR', 'DL']
save_dir = './figures2'

level_window_torch = lambda x, level, window: torch.clamp((x - level + window / 2) / window, 0, 1)
level_window = lambda x, level, window: np.clip((x - level + window / 2) / window, 0, 1)
level, window = 0, 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_dicom(dicom_dir):
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
    print(ds.RescaleSlope, ds.RescaleIntercept)

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

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for scanner in scanners:
        doses_files = []
        for dose in doses:
            dose_files = sorted(glob(os.path.join(data_dir, scanner, f'*{dose}*{reconstruction_method}*')))
            dose_files = [dose_file for dose_file in dose_files if os.path.isdir(dose_file)]
            doses_files.append(dose_files)

        reference_dose = doses_files[-1]
        doses_files = doses_files[:-1]
        plot_ref = True
        average_psrns, average_ssim, average_rmse = [], [], []
        for dose_idx, dose_files in enumerate(doses_files):
            plot_dose = True
            psrns, ssims, rmses = [], [], []
            for file_idx, dose_file in enumerate(dose_files):
                img = read_dicom(dose_file)
                try:
                    gt = read_dicom(reference_dose[file_idx])
                except:
                    #print(f'GT not found for {dose_file}')
                    continue
                img, _, _ = find_shifts(img, gt)
                if plot_ref:
                    plot_ref = False
                    plot_save(gt, f'./figures/{scanner}_14mGy_ref.png')
                if plot_dose:
                    plot_dose = False
                    plot_save(img, f'./figures/{scanner}_{doses[dose_idx]}.png')
                _psnr = psnr(img, gt).item()
                _ssim = ssim(img, gt).item()
                _rmse = rmse(img, gt).item()                    
                psrns.append(_psnr)
                ssims.append(_ssim)
                print(_rmse)
                rmses.append(_rmse)
            if len(psrns) == 0:
                psrns.append([-1, -1, -1])
                ssims.append([-1, -1, -1])
                rmses.append([-1, -1, -1])
            print(f'{scanner} {doses[dose_idx]} RMSE: {np.median(np.array(rmses)):.4f}+{np.array(rmses).std():.4f}, PSNR: {np.median(np.array(psrns)):.4f}+{np.array(psrns).std():.4f}, SSIM: {np.median(np.array(ssims)):.4f}+{np.array(ssims).std():.4f}')
            [psrns, ssims, rmses] = [np.array(item) for item in [psrns, ssims, rmses]]
            average_psrns.append(np.median(psrns))
            std_psrn = psrns.std()
            average_ssim.append(np.median(ssims))
            std_ssim = ssims.std()
            average_rmse.append(np.median(rmses))
            std_rmse = rmses.std()
        [average_rmse, average_psrns, average_ssim] = [np.array(item) for item in [average_rmse, average_psrns, average_ssim]]
        [std_rmse, std_psrn, std_ssim] = [np.array(item) for item in [std_rmse, std_psrn, std_ssim]]
        print(f'{scanner}   Average RMSE: {average_rmse.mean():.4f}+{average_rmse.std():.4f}, Average PSNR: {average_psrns.mean():.4f}+{average_psrns.std():.4f}, Average SSIM: {average_ssim.mean():.4f}+{average_ssim.std():.4f}')

if __name__ == '__main__':
    main()