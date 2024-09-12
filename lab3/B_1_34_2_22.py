
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
import sys
import numpy as np
import matplotlib.pyplot as plt
import time


class Convolution3D(nn.Module):
    def __init__(self, mask):
        super(Convolution3D, self).__init__()
        self.mask = mask

    def forward2(self, input_images):
        # Convert input images to grayscale
        input_images_gray = input_images.mean(dim=1, keepdim=True)
        print(input_images_gray.shape)
        # plt.imshow(input_images_gray)
        output_images = F.conv2d(input_images, self.mask, padding=2)  # Adjust padding as needed
        return output_images.sum(dim=1, keepdim=True)  # Sum across the channels d
    def forward1(self, input_images):
        output_images = []
        for channel in range(input_images.size(1)):  # Loop over each channel
            input_channel = input_images[:, channel:channel+1, :, :]  # Select single channel
            output_channel = F.conv2d(input_channel, self.mask, padding=2)  # Apply convolution
            output_images.append(output_channel)
        output_images = torch.cat(output_images, dim=1)  # Concatenate along the channel dimension
        return output_images.sum(dim=1, keepdim=True)  # Sum across the channels to get single-channel output
    def forward(self, input_images):
        print("dim of input_images", input_images.size(1))
        output_images = []
        padding = (self.mask.shape[2] // 2, self.mask.shape[3] // 2)  # Calculate padding for height and width
        
        for channel in range(input_images.size(1)):  # Loop over each channel
            input_channel = input_images[:, channel:channel+1, :, :]  # Select single channel
            output_channel = F.conv2d(input_channel, self.mask, padding=padding)  # Apply convolution with padding to maintain input size
            output_images.append(output_channel)
        output_images = torch.cat(output_images, dim=1)  # Concatenate along the channel dimension
        return output_images.sum(dim=1, keepdim=True)  # Sum across the channels to get single-channel output
    def forward6(self, input_images):
        print("dim of input_images", input_images.size())
        output_images = []
        padding = (self.mask.shape[2] // 2, self.mask.shape[3] // 2)  # Calculate padding for height and width
        for channel in range(input_images.size(1)):  # Loop over each channel
            input_channel = input_images[:, channel:channel+1, :, :]  # Select single channel
            output_channel = F.conv2d(input_channel, self.mask, padding=padding, padding_mode='replicate')  # Apply convolution with padding to maintain input size
            output_images.append(output_channel)
        output_images = torch.cat(output_images, dim=1)  # Concatenate along the channel dimension
        return output_images
def load_mask(mask_file):
    with open(mask_file, 'r') as f:
        mask_data = f.readlines()
        mask_size = int(mask_data[0])
        mask_values = [[float(val) for val in line.split()] for line in mask_data[1:]]
        mask_tensor = torch.tensor(mask_values).unsqueeze(0).unsqueeze(0)  # Single-channel mask
    return mask_tensor

def main(input_folder, output_folder, batch_size, mask_file):
    # Load mask
    mask = load_mask(mask_file)

    # Read input images from folder
    input_images = []
    for filename in os.listdir(input_folder)[:batch_size]:
        try:
            img = Image.open(os.path.join(input_folder, filename)).convert('RGB')
            img_tensor = torch.tensor(np.array(img)).unsqueeze(0).permute(0, 3, 1, 2).float()  # Convert image to tensor with dtype float32
            input_images.append(img_tensor)
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    # Stack input images into a single tensor
    input_images = torch.cat(input_images, dim=0)

    # Initialize 3D convolution module
    convolution_3d = Convolution3D(mask)

    # Apply convolution
    # output_images = convolution_3d(input_images)

    start_time = time.time()

    # Apply convolution
    print("input_images", input_images.shape)
    output_images = convolution_3d(input_images)
    print("output_images", output_images.shape)

    # Calculate execution time
    execution_time = time.time() - start_time
    print("Convolution execution time:", execution_time / 3, "seconds")
    # in ms
    print("Convolution execution time:", execution_time*1000 /3, "ms")
    # Save output images
    for i, output_image in enumerate(output_images):
        output_image = output_image.squeeze().numpy().astype('uint8')
        Image.fromarray(output_image).save(os.path.join(output_folder, f'output_{i}.png'))


if __name__ == "__main__":
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    batch_size = int(sys.argv[3])
    mask_file = sys.argv[4]
    # input_folder = "img2"
    # output_folder ="out"
    # batch_size = 1
    # mask_file = "mask9.txt"
    main(input_folder, output_folder, batch_size, mask_file)
