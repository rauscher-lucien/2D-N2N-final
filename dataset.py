import os
import torch
import tifffile
import numpy as np

class TwoVolumeDataset(torch.utils.data.Dataset):
    def __init__(self, volume_folder_paths, transform=None, num_volumes=2):
        """
        Args:
            volume_folder_paths (list): List of paths to folders containing volumes.
            transform (callable, optional): Optional transform to be applied on a sample.
            num_volumes (int, optional): Maximum number of volumes to consider in each folder.
        """
        self.volume_folder_paths = volume_folder_paths
        self.transform = transform
        self.num_volumes = num_volumes
        self.slice_pairs = self._preload_slice_pairs()

    def _preload_slice_pairs(self):
        """
        Preloads all possible slice pairs across the first `n` volumes in each folder without loading volumes into memory.
        
        Returns:
            list: A list of tuples, each containing the paths of two volumes and the slice index.
        """
        slice_pairs = []
        for folder in self.volume_folder_paths:
            volume_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.tiff')])

            # Limit to the first `n` volumes
            volume_files = volume_files[:min(len(volume_files), self.num_volumes)]

            if len(volume_files) > 1:
                for i in range(len(volume_files) - 1):  # Consider adjacent volumes only
                    volume_1 = volume_files[i]
                    volume_2 = volume_files[i + 1]

                    # Load the header of each volume to determine the number of slices without loading full data
                    volume_1_slices = tifffile.TiffFile(volume_1).pages
                    volume_2_slices = tifffile.TiffFile(volume_2).pages

                    # Take the minimum number of slices to avoid out-of-bounds issues
                    num_slices = min(len(volume_1_slices), len(volume_2_slices))

                    # Generate pairs of slice indices for these two volumes
                    for slice_idx in range(num_slices):
                        slice_pairs.append((volume_1, volume_2, slice_idx))

        return slice_pairs

    def __len__(self):
        """
        Returns the number of slice pairs, which equals the total number of possible pairs across all volumes.
        """
        return len(self.slice_pairs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the sample.
        
        Returns:
            tuple: (input_slice, target_slice)
        """
        # Retrieve the volume paths and slice index for this pair
        volume_1_path, volume_2_path, slice_idx = self.slice_pairs[index]

        # Load only the specific slices from the volumes
        volume_1_slice = tifffile.imread(volume_1_path, key=slice_idx)
        volume_2_slice = tifffile.imread(volume_2_path, key=slice_idx)

        input_slice = volume_1_slice
        target_slice = volume_2_slice

        # Apply transformations if specified
        if self.transform:
            input_slice, target_slice = self.transform((input_slice, target_slice))

        # Add channel dimension
        input_slice = input_slice[np.newaxis, ...]
        target_slice = target_slice[np.newaxis, ...]

        return input_slice, target_slice




class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, transform=None):
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.preloaded_data = {}  # To store preloaded data
        self.slices = self.preload_and_make_slices(root_folder_path)

    def preload_and_make_slices(self, root_folder_path):
        slices = []
        for subdir, _, files in os.walk(root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith(('.tif', '.tiff'))])
            for f in sorted_files:
                full_path = os.path.join(subdir, f)
                volume = tifffile.imread(full_path)
                self.preloaded_data[full_path] = volume  # Preload data here
                num_slices = volume.shape[0]
                for i in range(num_slices):  # Include all slices
                    slices.append((full_path, i))
        return slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        file_path, slice_index = self.slices[index]
        
        # Access preloaded data instead of reading from file
        input_slice = self.preloaded_data[file_path][slice_index]

        if self.transform:
            input_slice = self.transform(input_slice)
        
        # Add extra channel axis at position 0
        input_slice = input_slice[np.newaxis, ...]

        return input_slice