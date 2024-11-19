# Create Figures from Registered dataset

import os
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import nibabel as nib
from skimage.transform import resize
from utils import flip_volume
from PIL import Image
# Data directory
registered_dataset_dir = '/mnt/nas7/data/reza/registered_dataset_v2'
scanners = ['A1', 'A2', 'B1', 'B2', 'C1', 'D1', 'E1', 'E2', 'F1', 'G1', 'G2', 'H1', 'H2']
out_dir = 'scanner_comparison_test'
patch_center_cords = [275-185,295]
patch_center_cords = [275-110,204]
patch_size, magnification = 32, 3
slice_idx = 130
level, window =  50, 400 # 30, 150 # 0, 1000 #
reconstruction = '*'#'IR' 'DL' 'FBP'

# Lambda functions
level_window = lambda x, level, window: np.clip((x - level + window / 2) / window, 0, 1)

def compare_scans(registered_dataset_dir, out_dir, scanners, reconstruction='IR'):
    # List files:
    all_files = glob(os.path.join(registered_dataset_dir, f'*{reconstruction}*.nii.gz'))
    scanners_files = {scanner: [f for f in all_files if f'{scanner}_' in f] for scanner in scanners}

    # Pick one file per scanner
    random_seed = 0
    files_to_compare = []
    for key in scanners_files.keys():
        np.random.seed(random_seed)
        files_to_compare.append(np.random.choice(scanners_files[key]))

    # Create the output directory
    os.makedirs(out_dir, exist_ok=True)
    
    for item in files_to_compare:
        print(item)
        # Load the data
        image = nib.load(item).get_fdata()
        image = image.transpose(1, 0, 2)
        image = flip_volume(image, axis=0)
        image = level_window(image, level, window)

        slice = image[..., slice_idx]
        # Create a three channel slice
        slice = np.stack([slice, slice, slice], axis=-1)

        patch_data = slice[patch_center_cords[0]-patch_size//2:patch_center_cords[0]+patch_size//2, patch_center_cords[1]-patch_size//2:patch_center_cords[1]+patch_size//2, :].copy()
        # Resize the patch data with a magnification factor
        patch_data_pil = Image.fromarray((patch_data*255).astype(np.uint8))
        patch_data_zoom = patch_data_pil.resize((patch_size*magnification, patch_size*magnification), Image.LANCZOS)
        patch_data_zoom = np.array(patch_data_zoom)/255
        #patch_data_zoom = resize(patch_data, (patch_size*magnification, patch_size*magnification), anti_aliasing=False)

        # # Create a red square at the patch center with patch size
        # slice[patch_center_cords[0]-patch_size//2-1:patch_center_cords[0]+patch_size//2+1, patch_center_cords[1]-patch_size//2-1:patch_center_cords[1]+patch_size//2+1, :] = [1, 0, 0]
        # slice[patch_center_cords[0]-patch_size//2:patch_center_cords[0]+patch_size//2, patch_center_cords[1]-patch_size//2:patch_center_cords[1]+patch_size//2, :] = patch_data

        # # Create a red square around the zoomed patch
        # slice[-patch_size*magnification-2:, 0:patch_size*magnification+2, :] = [1, 0, 0]
        # # Embed the patch data zommed in the left down corner of the slice 
        # slice[-patch_size*magnification-1:-1, 1:patch_size*magnification+1, :] = patch_data_zoom
        

        # Plot the slice
        plt.imshow(slice, 'gray')
        plt.axis('off')
        plt.savefig(os.path.join(out_dir, os.path.basename(item) + '.png'), bbox_inches='tight', pad_inches=0)
        #plt.show()
        plt.close()

if __name__ == '__main__':
    compare_scans(registered_dataset_dir, out_dir, scanners, reconstruction=reconstruction)