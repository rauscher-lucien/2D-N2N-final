import os
import numpy as np
import torch
import tifffile
import pickle


def get_device():
    if torch.cuda.is_available():
        print("GPU is available")
        device = torch.device("cuda:0")
    else:
        print("GPU is not available")
        device = torch.device("cpu")
    
    return device



def load_volume_folder_paths(txt_file_path):
    """
    Reads a text file containing paths to folders (one per line) and returns a list of those paths,
    stripping any extra quotes around the paths.

    Args:
        txt_file_path (str): The path to the text file containing folder paths.

    Returns:
        list: A list of folder paths.
    """
    if not os.path.exists(txt_file_path):
        raise FileNotFoundError(f"The specified file does not exist: {txt_file_path}")

    with open(txt_file_path, 'r') as file:
        folder_paths = [line.strip().strip('"') for line in file]

    # Optionally, you can add filtering to ignore empty lines or comments
    folder_paths = [path for path in folder_paths if path and not path.startswith("#")]

    return folder_paths




def create_result_dir(project_dir, project_name, hyperparameters):
    # Create a name based on project name and hyperparameters
    hyperparams_str = '_'.join([f"{key}{value}" for key, value in hyperparameters.items()])
    name = f"{project_name}_{hyperparams_str}"

    results_dir = os.path.join(project_dir, name, 'results')
    os.makedirs(results_dir, exist_ok=True)
    checkpoints_dir = os.path.join(project_dir, name, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    return results_dir, checkpoints_dir



def create_train_dir(results_dir):
    train_dir = os.path.join(results_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)

    return train_dir



def compute_global_mean_and_std(volume_folders, checkpoints_path):
    """
    Computes and saves the global mean and standard deviation across all TIFF stacks
    in the given directories, saving the results in the checkpoints directory.

    Parameters:
    - volume_folders: List of paths to directories containing the TIFF files.
    - checkpoints_path: Path to the directory where the normalization parameters will be saved.
    """
    # Define the save_path in the checkpoints directory
    save_path = os.path.join(checkpoints_path, 'normalization_params.pkl')

    # Check if the normalization_params.pkl file already exists
    if os.path.exists(save_path):
        # Load the existing mean and std values
        with open(save_path, 'rb') as f:
            params = pickle.load(f)
            global_mean = params['mean']
            global_std = params['std']
        print(f"Loaded global mean and std parameters from {save_path}")
    else:
        # First, count the total number of TIFF files
        total_files = sum(
            len([filename for filename in files if filename.lower().endswith(('.tif', '.tiff'))])
            for folder in volume_folders
            for _, _, files in os.walk(folder)
        )
        print(f"Total number of TIFF files to process: {total_files}")

        all_means = []
        all_stds = []
        file_counter = 0

        for folder in volume_folders:
            for subdir, _, files in os.walk(folder):
                for filename in files:
                    if filename.lower().endswith(('.tif', '.tiff')):
                        filepath = os.path.join(subdir, filename)
                        print(f"Processing volume {file_counter + 1}/{total_files}: {filepath}")
                        
                        stack = tifffile.imread(filepath)
                        all_means.append(np.mean(stack))
                        all_stds.append(np.std(stack))
                        
                        file_counter += 1

        global_mean = np.mean(all_means)
        global_std = np.mean(all_stds)
        
        # Save the computed global mean and standard deviation to a file
        with open(save_path, 'wb') as f:
            pickle.dump({'mean': global_mean, 'std': global_std}, f)
        
        print(f"Global mean and std parameters saved to {save_path}")

    return global_mean, global_std


