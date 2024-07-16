# Code for registration of all images
import itk
import ants
import numpy as np
from scipy.ndimage import zoom
from utils import read_dicom

class ImageRegistrator:
    def __init__(self, registration_method, reference_image=None, params=None):
        # Initialize any necessary variables or data structures
        self.registration_method = registration_method
        self.reference_image = reference_image
        if 'elastic' in self.registration_method:
            # Define the defualt parameters of the elastix registration:
            self.parameter_object = itk.ParameterObject.New()
            self.default_parameter_map = parameter_object.GetDefaultParameterMap('rigid')
            self.default_parameter_map['MaximumNumberOfIterations'] = ['1000']
            self.parameter_object.AddParameterMap(default_parameter_map)
            # Overwrite the default parameters with the user-defined parameters
            if params is not None:
                for key in params:
                    self.default_parameter_map[key] = params[key]
            self.register_image = self.register_image_elastic

        elif 'ants' in self.registration_method:
            self.register_image = self.register_image_ants

    def get_image(self, image):
        if isinstance(image, str):
            # Load the image from file
            image = read_dicom(image)
        elif isinstance(image, np.ndarray):
            # Convert the image array to ITK image
            image = image
        return image
    
    def resample_image(self, volume, factor=4):
        # Upsample the image by a factor of 4
        zoom_factors = (factor, factor, factor)
        volume = zoom(volume, zoom_factors, order=3) 
        return volume

    def register_image_elastic(self, image, reference_image, downsample_factor=4):
        # Register the image using elastix registration
        if self.referece_image is None:
            if reference_image is None:
                raise ValueError('Reference image is required!')
            else:
                fixed_image = self.get_image(reference_image)
        else:
            fixed_image = self.reference_image
        
        # Downsample the images by a factor of 4
        fixed_image_down = self.resample_image(fixed_image, downsample_factor)
        moving_image_down = self.resample_image(moving_image, downsample_factor)
        
        # Convert the moving image array to ITK image
        fixed_image = itk.GetImageFromArray(fixed_image_down)
        moving_image = itk.GetImageFromArray(moving_image_down)
        
        # Load Elastix Image Filter Object
        elastix_object = itk.ElastixRegistrationMethod.New(fixed_image_down, moving_image_down)
        elastix_object.SetParameterObject(self.parameter_object)
        
        # Set additional options
        elastix_object.SetLogToConsole(False)

        # Update filter object (required)
        elastix_object.UpdateLargestPossibleRegion()
        
        # Computer the Registration Transformation
        self.result_transform_parameters = elastix_object.GetTransformParameterObject()
        
        # Apply the transform to the moving image
        result_image = elastix_object.GetOutput()
        
        # Upsample the image to the original size
        image_registered = self.resample(itk.GetArrayFromImage(result_image), 1/downsample_factor)

        return image_registered, reference_image
    
    def register_image_ants(self, image, reference_image, downsample_factor=4):

        # Register the image using elastix registration
        if self.referece_image is None:
            if reference_image is None:
                raise ValueError('Reference image is required!')
            else:
                fixed_image = self.get_image(reference_image)
        else:
            fixed_image = self.reference_image
        moving_image = self.get_image(image)
        
        # Downsample the images by a factor of 4
        fixed_image_down = self.resample_image(fixed_image, downsample_factor)
        moving_image_down = self.resample_image(moving_image, downsample_factor)
        
        # Register the images
        registration_outs = ants.registration(fixed=fixed_image_down, moving=moving_image_down, type_of_transform = 'SyN' )
        
        # Get the registered images
        _, image_registered = registration_outs['warpedfixout'], registration_outs['warpedmovout']

        # Upsample the image to the original size
        image_registered = self.resample_image(image_registered, 1/downsample_factor)
        
        return image_registered, reference_image