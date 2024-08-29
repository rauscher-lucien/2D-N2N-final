import numpy as np
import torch



class Normalize(object):
    """
    Normalize an image using mean and standard deviation.
    
    Args:
        mean (float or tuple): Mean for each channel.
        std (float or tuple): Standard deviation for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        """
        Args:
            data (tuple): Containing input and target images to be normalized.
        
        Returns:
            Tuple: Normalized input and target images.
        """
        input_img, target_img = data

        # Normalize input image
        input_normalized = (input_img - self.mean) / self.std

        # Normalize target image
        target_normalized = (target_img - self.mean) / self.std

        return input_normalized, target_normalized
    


    
class RandomCrop:
    def __init__(self, output_size=(64, 64)):
        """
        RandomCrop constructor for cropping both the input stack of slices and the target slice.
        Args:
            output_size (tuple): The desired output size (height, width).
        """
        self.output_size = output_size

    def __call__(self, data):
        """
        Apply the cropping operation.
        Args:
            data (tuple): A tuple containing the input stack and the target slice.
        Returns:
            Tuple: Cropped input stack and target slice.
        """
        input_stack, target_slice = data

        h, w = input_stack.shape
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        input_cropped = input_stack[top:top+new_h, left:left+new_w]
        target_cropped = target_slice[top:top+new_h, left:left+new_w]

        return input_cropped, target_cropped
    


    
class RandomHorizontalFlip:
    def __call__(self, data):
        """
        Apply random horizontal flipping to both the input stack of slices and the target slice.
        In 50% of the cases, only horizontal flipping is applied without vertical flipping.
        
        Args:
            data (tuple): A tuple containing the input stack and the target slice.
        
        Returns:
            Tuple: Horizontally flipped input stack and target slice, if applied.
        """
        input_stack, target_slice = data

        # Apply horizontal flipping with a 50% chance
        if np.random.rand() > 0.5:
            # Flip along the width axis (axis 1), keeping the channel dimension (axis 2) intact
            input_stack = np.flip(input_stack, axis=1)
            target_slice = np.flip(target_slice, axis=1)

        # With the modified requirements, we remove the vertical flipping part
        # to ensure that only horizontal flipping is considered.

        return input_stack, target_slice
    


    
class ToTensor(object):
    def __call__(self, data):
        def convert_image(img):
            return torch.from_numpy(img.astype(np.float32))
        return tuple(convert_image(img) for img in data)
    



class ToNumpy(object):

    def __call__(self, data):

        return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    



class Denormalize(object):
    """
    Denormalize an image using mean and standard deviation, then convert it to 16-bit format.
    
    Args:
        mean (float or tuple): Mean for each channel.
        std (float or tuple): Standard deviation for each channel.
    """

    def __init__(self, mean, std):
        """
        Initialize with mean and standard deviation.
        
        Args:
            mean (float or tuple): Mean for each channel.
            std (float or tuple): Standard deviation for each channel.
        """
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        """
        Denormalize the image and convert it to 16-bit format.
        
        Args:
            img (numpy array): Normalized image.
        
        Returns:
            numpy array: Denormalized 16-bit image.
        """
        # Denormalize the image by reversing the normalization process
        img_denormalized = (img * self.std) + self.mean

        # Scale the image to the range [0, 65535] and convert to 16-bit unsigned integer
        img_16bit = img_denormalized.astype(np.uint16)
        
        return img_16bit