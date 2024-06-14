# Computer PSNRs for Dose
import os
import torch
import pydicom
import numpy as np
from glob import glob
from monai import metrics

data_dir = '/mnt/nas4/datasets/ToCurate/QA4IQI/FinalDataset-TCIA-MultiCentric/Upl'
scanners = ['A1', 'A2', 'B1', 'B2', 'C1', 'D1', 'E1', 'E2', 'F1', 'G1', 'G2', 'H1', 'H2']
doses = ['1mGy', '3mGy', '6mGy', '10mGy', '14mGy']
reconstruction_method = ['FBP', 'IR', 'DL']

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

    # Now 'volume' contains the 3D volume of the DICOM slices
    # print(volume.shape)
    return torch.tensor(volume.transpose(1, 2, 0)).to(device).float().unsqueeze(0).unsqueeze(0)

# Main Fucntion
def main():
    psnr = metrics.PSNRMetric(max_val=2000)
    ssim = metrics.SSIMMetric(spatial_dims=3, data_range=2000)
    rmse = metrics.RMSEMetric()

    average_psrns, average_ssim = [], []
    for scanner in scanners:
        doses_files = []
        for dose in doses:
            dose_files = sorted(glob(os.path.join(data_dir, scanner, f'*{dose}*')))
            dose_files = [dose_file for dose_file in dose_files if os.path.isdir(dose_file)]
            doses_files.append(dose_files)

        psrns, ssims = [], []
        reference_dose = doses_files[-1]
        doses_files = doses_files[:-1]
        for dose_idx, dose_files in enumerate(doses_files):
            for file_idx, dose_file in enumerate(dose_files):
                img = read_dicom(dose_file)
                gt = read_dicom(reference_dose[file_idx])
                _psnr = psnr(img, gt).item()
                _ssim = ssim(img, gt).item()
                psrns.append(_psnr)
                ssims.append(_ssim)
            print(f'{scanner} {doses[dose_idx]} PSNR: {np.array(psrns).mean():.2f} SSIM: {np.array(ssims).mean():.2f}')
            average_psrns.append(np.array(psrns).mean())
            average_ssim.append(np.array(ssims).mean())
        print(f'{scanner} Average PSNR: {np.array(average_psrns).mean()} Average SSIM: {np.array(average_ssim).mean()}')

if __name__ == '__main__':
    main()