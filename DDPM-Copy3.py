#!/usr/bin/env python
# coding: utf-8

# In[1]:


import diffusers as dif
from diffusers import UNet2DConditionModel
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
import os
from dataclasses import dataclass
import h5py

class PatientCTCBCTDataset(Dataset):
    def __init__(self, dataset_path, transform=None, dynamic_scaling=False):
        self.dataset_path = dataset_path
        self.patients_list = []  # List to keep track of patients
        self.transform = transform  # Transformation to be applied to each tensor
        self.dynamic_scaling = dynamic_scaling  # Option to dynamically scale data between -1 and 1

        # Open the HDF5 file and populate patients_list
        with h5py.File(self.dataset_path, 'r') as file:
            for patient in file.keys():
                # Ensure both CT and CBCT data are present for the patient
                if 'CT' in file[patient] and 'CBCT' in file[patient]:
                    self.patients_list.append(patient)

    def __len__(self):
        # Total number of patients with both CT and CBCT data
        return len(self.patients_list)

    def __getitem__(self, idx):
        patient = self.patients_list[idx]

        # Load both CT and CBCT data for the specified patient
        with h5py.File(self.dataset_path, 'r') as file:
            ct_data = file[patient]['CT'][()]
            cbct_data = file[patient]['CBCT'][()]

        # Convert data to PyTorch tensors and ensure the data type is float
        ct_tensor = torch.tensor(ct_data, dtype=torch.float)
        cbct_tensor = torch.tensor(cbct_data, dtype=torch.float)

        if self.dynamic_scaling:
            # Dynamically scale the data between -1 and 1
            ct_tensor = 2 * ((ct_tensor - ct_tensor.min()) / (ct_tensor.max() - ct_tensor.min())) - 1
            cbct_tensor = 2 * ((cbct_tensor - cbct_tensor.min()) / (cbct_tensor.max() - cbct_tensor.min())) - 1

        if self.transform:
            # Apply the transformation to both CT and CBCT tensors
            ct_tensor = self.transform(ct_tensor)
            cbct_tensor = self.transform(cbct_tensor)

        return ct_tensor, cbct_tensor
# In[45]:


@dataclass
class TrainingConfig:
    image_size = 64  # the generated image resolution
    train_batch_size = 64
    eval_batch_size = 64  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-5
    lr_warmup_steps = 200
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "cbct_ct_ddpm4.3"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


config = TrainingConfig()


# In[46]:


dataset_path= "/opt/users/dpac0031/training/3CT_CBCT.hdf5"
device=torch.device("cuda")


# In[47]:




# Path to the HDF5 file
hdf5_file_path = '/opt/users/dpac0031/training/3CT_CBCT.hdf5'

# Open the HDF5 file
with h5py.File(hdf5_file_path, 'r') as hdf5_file:
    print(f"File: {hdf5_file_path}")
    print(f"Groups/Datasets in the file:")

    # Function to recursively print the structure of the HDF5 file
    def print_structure(name, obj):
        print(f"{name}: {'Group' if isinstance(obj, h5py.Group) else 'Dataset'}")
        if isinstance(obj, h5py.Dataset):
            print(f"  - Shape: {obj.shape}")
            print(f"  - Dtype: {obj.dtype}")
            print(f"  - Compression: {obj.compression}")
            for attr in obj.attrs:
                print(f"  - Attribute: {attr}, Value: {obj.attrs[attr]}")

    # Recursively walk through the file and print structure
    hdf5_file.visititems(print_structure)


# In[ ]:


PatientCTCBCTDataset()


# In[48]:


dataset= PatientCTCBCTDataset(dataset_path,dynamic_scaling=True)
BATCH_SIZE=64


# In[49]:


noise_scheduler = DDPMScheduler(num_train_timesteps=1000)


# In[50]:


from diffusers import UNet2DModel

model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=4,  # the number of input channels, 3 for RGB images
    out_channels=2,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(64, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "AttnDownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
    ),
)


# In[51]:


loader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)


# In[52]:


def plot_image_channels(dataset, num_images=3):
    """
    Plots the two channels for each image in the dataset separately.

    Parameters:
    - dataset: The dataset object containing the images.
    - num_images: The number of images to plot from the dataset.
    """

    # Ensure we don't try to plot more images than the dataset contains
    num_images = min(num_images, len(dataset))

    for i in range(num_images):
        # Get the CT and CBCT tensors for the i-th patient
        ct_tensor, cbct_tensor = dataset[i]
        sample_image=ct_tensor
        noise = torch.randn(sample_image.shape)
        timesteps = torch.LongTensor([0])
        noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)
        # There are 2 channels in each tensor, so we plot each channel separately
        for channel in range(2):
            plt.figure(figsize=(10, 4))

            # Plot CT channel
            plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
            plt.imshow(noisy_image[channel].numpy(), cmap='gray')
            plt.title(f'Patient {i+1} - CT Channel {channel+1}')
            plt.axis('off')  # Hide the axis

            # Plot CBCT channel
            plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
            plt.imshow(cbct_tensor[channel].numpy(), cmap='gray')
            plt.title(f'Patient {i+1} - CBCT Channel {channel+1}')
            plt.axis('off')  # Hide the axis

            plt.show()


plot_image_channels(dataset, num_images=3)


# In[58]:


num_epochs=20
learning_rate=1e-5
lr_warmup_steps=20


# In[55]:


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=lr_warmup_steps,
    num_training_steps=(len(loader) * num_epochs),
)

# In[57]:


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    for epoch in tqdm(range(config.num_epochs)):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process,leave=False)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch[0]
            condition = batch[1]



            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            conc_input = torch.cat((noisy_images, condition), 1)

            with accelerator.accumulate(model):
                noise_pred = model(conc_input, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # Save the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            model_path = os.path.join(config.output_dir, f"model_epoch_{epoch + 1}.pth")
            accelerator.save(model.state_dict(), model_path)

    # You can also save the final model outside the loop if needed
    final_model_path = os.path.join(config.output_dir, "final_model.pth")
    accelerator.save(model.state_dict(), final_model_path)




from accelerate import notebook_launcher

args = (config, model, noise_scheduler, optimizer, loader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)








