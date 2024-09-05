import os
import torch
import matplotlib.pyplot as plt
import pickle
import time
import logging
from tqdm import tqdm

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from utils import *
from transforms import *
from dataset import *
from model import *


class Trainer:
    def __init__(self, data_dict):
        self.volume_folder_paths = data_dict['volume_folder_paths']  # List of paths to folders containing volumes
        self.project_dir = data_dict['project_dir']
        self.project_name = data_dict['project_name']

        self.disp_freq = data_dict['disp_freq']
        self.train_continue = data_dict['train_continue']

        self.hyperparameters = data_dict['hyperparameters']

        self.model_name = self.hyperparameters['model_name']
        self.UNet_base = self.hyperparameters['UNet_base']
        self.num_epoch = self.hyperparameters['num_epoch']
        self.batch_size = self.hyperparameters['batch_size']
        self.lr = self.hyperparameters['lr']
        self.patience = self.hyperparameters.get('patience', 10)  # Load patience with a default value

        self.device = get_device()

        self.volume_folders = load_volume_folder_paths(self.volume_folder_paths)

        # Create result and checkpoint directories using only project information and hyperparameters
        self.results_dir, self.checkpoints_dir = create_result_dir(
            self.project_dir, self.project_name, self.hyperparameters)
        self.train_results_dir = create_train_dir(self.results_dir)

        self.writer = SummaryWriter(self.results_dir + '/tensorboard_logs')

        # Compute mean and std during initialization
        start_time = time.time()
        self.mean, self.std = compute_global_mean_and_std(self.volume_folders, self.checkpoints_dir)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Mean and std computation execution time: {execution_time} seconds")

    def save(self, checkpoints_dir, model, optimizer, epoch, best_train_loss):
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_train_loss': best_train_loss,
            'hyperparameters': self.hyperparameters
        }, os.path.join(checkpoints_dir, 'best_model.pth'))

    def load(self, checkpoints_dir, model, device, optimizer):
        checkpoint_path = os.path.join(checkpoints_dir, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        dict_net = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(dict_net['model'])
        optimizer.load_state_dict(dict_net['optimizer'])
        epoch = dict_net['epoch']
        best_train_loss = dict_net.get('best_train_loss', float('inf'))
        self.hyperparameters = dict_net.get('hyperparameters', self.hyperparameters)

        print(f'Loaded {epoch}th network with hyperparameters: {self.hyperparameters}, best train loss: {best_train_loss:.4f}')

        return model, optimizer, epoch, best_train_loss

    def get_model(self):
        if self.model_name == 'UNet3':
            return UNet3(base=self.UNet_base).to(self.device)
        elif self.model_name == 'UNet4':
            return UNet4(base=self.UNet_base).to(self.device)
        elif self.model_name == 'UNet5':
            return UNet5(base=self.UNet_base).to(self.device)
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")



    def train(self):
        save_interval_batches = 100  # Check loss and save model every 100 batches (adjust as needed)
        progress_update_interval = 10  # Update progress bar every 10 batches

        transform_train = transforms.Compose([
            Normalize(self.mean, self.std),
            RandomCrop(output_size=(64,64)),
            RandomHorizontalFlip(),
            ToTensor()
        ])

        transform_inv_train = transforms.Compose([
            ToNumpy(),
            Denormalize(self.mean, self.std)
        ])

        ### make dataset and loader ###
        dataset_train = TwoVolumeDataset(self.volume_folders, transform_train)

        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

        ### initialize network ###
        model = self.get_model()
        criterion = nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), self.lr)

        st_epoch = 0
        best_train_loss = float('inf')
        patience_counter = 0  # Initialize patience counter

        if self.train_continue == 'on':
            print(self.checkpoints_dir)
            model, optimizer, st_epoch, best_train_loss = self.load(self.checkpoints_dir, model, self.device, optimizer)
            model = model.to(self.device)

        for epoch in range(st_epoch + 1, self.num_epoch + 1):
            model.train()  # Ensure model is in training mode
            train_loss = 0.0

            # Add a tqdm progress bar for the epoch
            with tqdm(total=len(loader_train), desc=f"Epoch {epoch}/{self.num_epoch}", unit="batch") as pbar:
                for batch, data in enumerate(loader_train, 1):
                    optimizer.zero_grad()
                    input_slice, target_img = [x.squeeze(0).to(self.device) for x in data]
                    output_img = model(input_slice)

                    loss = criterion(output_img, target_img)
                    train_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                    # Update the progress bar every `progress_update_interval` batches
                    if batch % progress_update_interval == 0 or batch == len(loader_train):
                        pbar.set_postfix({"Batch Loss": loss.item()})
                        pbar.update(progress_update_interval)

                    # Check and save model if the loss has improved every `save_interval_batches`
                    if batch % save_interval_batches == 0 or batch == len(loader_train):
                        avg_train_loss = train_loss / batch

                        if avg_train_loss < best_train_loss:
                            best_train_loss = avg_train_loss
                            self.save(self.checkpoints_dir, model, optimizer, epoch, best_train_loss)
                            patience_counter = 0  # Reset patience counter
                            print(f"Saved best model at epoch {epoch}, batch {batch} with loss {best_train_loss:.4f}")
                        else:
                            #patience_counter += 1  # Increment patience counter
                            print(f'Patience Counter: {patience_counter}/{self.patience}')

                        # Save images for visual inspection (only when saving model)
                        input_img = transform_inv_train(input_slice)[..., 0]
                        target_img = transform_inv_train(target_img)[..., 0]
                        output_img = transform_inv_train(output_img)[..., 0]

                        for j in range(target_img.shape[0]):
                            plt.imsave(os.path.join(self.train_results_dir, f"input_{j}.png"), input_img[j, :, :], cmap='gray')
                            plt.imsave(os.path.join(self.train_results_dir, f"target_{j}.png"), target_img[j, :, :], cmap='gray')
                            plt.imsave(os.path.join(self.train_results_dir, f"output_{j}.png"), output_img[j, :, :], cmap='gray')

                # Check for early stopping based on patience
                if patience_counter >= self.patience:
                    print(f'Early stopping triggered after {epoch} epochs')
                    break

        self.writer.close()