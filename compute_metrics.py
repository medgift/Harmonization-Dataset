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
scanners = ['A1', 'A2', 'B1', 'B2', 'C1', 'D1', 'E1', 'E2', 'F1', 'G1', 'G2', 'H1', 'H2']
doses = ['1mGy', '3mGy', '6mGy', '10mGy', '14mGy']
reconstruction_method = ['FBP', 'IR', 'DL']

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

# Main Fucntion
def main():
    psnr = metrics.PSNRMetric(max_val=2000)
    ssim = metrics.SSIMMetric(spatial_dims=3, data_range=2000)
    rmse = metrics.RMSEMetric()

    if not os.path.exists('./figures'):
        os.makedirs('./figures')

    for scanner in scanners:
        doses_files = []
        for dose in doses:
            dose_files = sorted(glob(os.path.join(data_dir, scanner, f'*{dose}*')))
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
                rmses.append(_rmse)
            print(f'{scanner} {doses[dose_idx]} RMSE: {np.array(rmses).mean():.4f}, PSNR: {np.array(psrns).mean():.4f}, SSIM: {np.array(ssims).mean():.4f}')
            average_psrns.append(np.array(psrns).mean())
            average_ssim.append(np.array(ssims).mean())
            average_rmse.append(np.array(rmses).mean())
        print(f'{scanner} Average RMSE: {np.array(average_rmse).mean():.4f}, Average PSNR: {np.array(average_psrns).mean():.4f}, Average SSIM: {np.array(average_ssim).mean():.4f}')

if __name__ == '__main__':
    main()