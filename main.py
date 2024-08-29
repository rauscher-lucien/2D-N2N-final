import os
import sys
import argparse
import logging

sys.path.append(os.path.join(".."))

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

logging.basicConfig(filename='logging.log',  # Log filename
                    filemode='a',  # Append mode, so logs are not overwritten
                    format='%(asctime)s - %(levelname)s - %(message)s',  # Include timestamp
                    level=logging.INFO,  # Logging level
                    datefmt='%Y-%m-%d %H:%M:%S')  # Timestamp format

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set logging level for console
logging.getLogger('').addHandler(console_handler)

# Redirect stdout and stderr to logging
sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)

from utils import *
from train import *

def main():
    # Check if the script is running on the server by looking for the environment variable
    if os.getenv('RUNNING_ON_SERVER') == 'true':
        parser = argparse.ArgumentParser(description='Process data directory.')

        parser.add_argument('--volume_folder_paths', type=str, help='Path to the text file listing folders containing volumes')
        parser.add_argument('--project_name', type=str, help='Name of the project')
        parser.add_argument('--train_continue', type=str, default='off', choices=['on', 'off'],
                            help='Flag to continue training: "on" or "off" (default: "off")')
        parser.add_argument('--disp_freq', type=int, default=10, help='Display frequency (default: 10)')
        parser.add_argument('--model_name', type=str, default='UNet3', help='Name of the model (default: UNet3)')
        parser.add_argument('--unet_base', type=int, default=32, help='Base number of filters in UNet (default: 32)')
        parser.add_argument('--num_epoch', type=int, default=1000, help='Number of epochs (default: 1000)')
        parser.add_argument('--batch_size', type=int, default=8, help='Batch size (default: 8)')
        parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate (default: 1e-5)')
        parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (default: 10)')

        args = parser.parse_args()

        volume_folder_paths = args.volume_folder_paths
        project_name = args.project_name
        train_continue = args.train_continue
        disp_freq = args.disp_freq
        model_name = args.model_name
        unet_base = args.unet_base
        num_epoch = args.num_epoch
        batch_size = args.batch_size
        lr = args.lr
        patience = args.patience
        project_dir = os.path.join('/g', 'prevedel', 'members', 'Rauscher', 'final_projects', '2D-N2N-final')

        print(f"Using volume folder paths: {volume_folder_paths}")
        print(f"Train continue: {train_continue}")
        print(f"Display frequency: {disp_freq}")
        print(f"Model name: {model_name}")
        print(f"UNet base: {unet_base}")
        print(f"Number of epochs: {num_epoch}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {lr}")
        print(f"Patience: {patience}")

    else:
        # Default settings for local testing
        volume_folder_paths = r"C:\Users\rausc\Documents\EMBL\final_projects\volume_folder_paths.txt"
        project_dir = r"C:\Users\rausc\Documents\EMBL\final_projects\2D-N2N-final"
        project_name = 'test_x'
        train_continue = 'on'
        disp_freq = 1
        model_name = 'UNet3'
        unet_base = 16
        num_epoch = 1000
        batch_size = 8
        lr = 1e-5
        patience = 10



    data_dict = {
        'volume_folder_paths': volume_folder_paths,
        'project_dir': project_dir,
        'project_name': project_name,
        'disp_freq': disp_freq,
        'train_continue': train_continue,
        'hyperparameters': {
            'model_name': model_name,
            'UNet_base': unet_base,
            'num_epoch': num_epoch,
            'batch_size': batch_size,
            'lr': lr,
            'patience': patience
        }
    }

    trainer = Trainer(data_dict)
    trainer.train()

if __name__ == '__main__':
    main()





