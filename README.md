# A Multi-Centric Anthropomorphic 3D CT Phantom-Based Benchmark Dataset for Harmonization

This repository contains all the tool to reproduce the Figures and Tables which are published in conjuction with this paper:

The dataset assosiated with theses analysis is publicly availbale here:

## The organization of the repository:
 There are two files (utils.py & ImageRegistration.py & registered_dataset.py) which contain the reuired class and functions for preprocessing the raw dicom dateset into Nifti format and the rest correspond to the codes for computing results and differnet measures presented in various Tables and Figures of the paper.

 ## Preprocessing the dataset:
Before computing various metircs in the image and feature domains or based on the predictions for diagnosis, we preprocess the data with the goal of removing the discripancies related to the position of the phantom in the scan to only take into the account diverences cause by the scanner used for image acquistion mainly reflected in the texture of the images.

The preprocessing also converts the original dicom stack into Nifti format which more straight forward to load for training deep learning model. These functions and classes are presented in the following scripts:
- registered_dataset.py: This script simply converts the original dicom dataset into registered NifTi dataset and crop the volumes into the region where phantom is and remove the air around the phantom.
- ImageRegistration: This class is developed for image registration and it contain two libraries to do the registration.
- utils.py: This files includes the function commonly uses in all the other scripts such as reading the data, fliping the volume through one axis, differnt data format conversions, and plots.


## Reprducing the Paper's results:
All the scripts developed for computing the metrics use the registered dataset in Nifti format for simplicity and enhancing the efficniency of loading the data. Therefore, prior to any other step the orginal dataset should be converted to Nifti format using registered_dataset.py script as described above.

Then, the resluts presented in the paper can be computed using the following codes:
- scanner_comparison.py: This is the script to depicted a slice of the phantom from the liver part and compare the texture properties of the scans acquired from differenct scanners. It is possible to zoom into a specific region and magnify it with an arbitrary factor for a close look into the texture changes (Figure 4). 
- compute_metrics_scanner.py: This code computes several metrics inlcuding root mean square (RMSE), peak singal to nosie ratio (PSNR) and structural similarty (SSIM) between differnet scanener and average it on several repititions of scan acquired using the harmonized protocol (Table 3-5).
- compute_metrics_dose.py: This code computes several metrics inlcuding root mean square (RMSE), peak singal to nosie ratio (PSNR) and structural similarty (SSIM) between differnet doses for each scanener and average it on several repititions of scan (Table 6).
