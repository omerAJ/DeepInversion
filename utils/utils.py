# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Pavlo Molchanov and Hongxu Yin
# --------------------------------------------------------

import torch
import os
from torch import distributed, nn
import random
import numpy as np

def load_model_pytorch(model, load_model, gpu_n=0):
    print("=> loading checkpoint '{}'".format(load_model))

    checkpoint = torch.load(load_model, map_location = lambda storage, loc: storage.cuda(gpu_n))

    if 'state_dict' in checkpoint.keys():
        load_from = checkpoint['state_dict']
    else:
        load_from = checkpoint

    if 1:
        if 'module.' in list(model.state_dict().keys())[0]:
            if 'module.' not in list(load_from.keys())[0]:
                from collections import OrderedDict

                load_from = OrderedDict([("module.{}".format(k), v) for k, v in load_from.items()])

        if 'module.' not in list(model.state_dict().keys())[0]:
            if 'module.' in list(load_from.keys())[0]:
                from collections import OrderedDict

                load_from = OrderedDict([(k.replace("module.", ""), v) for k, v in load_from.items()])

    if 1:
        if list(load_from.items())[0][0][:2] == "1." and list(model.state_dict().items())[0][0][:2] != "1.":
            load_from = OrderedDict([(k[2:], v) for k, v in load_from.items()])

        load_from = OrderedDict([(k, v) for k, v in load_from.items() if "gate" not in k])

    model.load_state_dict(load_from, strict=True)

    epoch_from = -1
    if 'epoch' in checkpoint.keys():
        epoch_from = checkpoint['epoch']
    print("=> loaded checkpoint '{}' (epoch {})"
        .format(load_model, epoch_from))


def create_folder(directory):
    # from https://stackoverflow.com/a/273227
    if not os.path.exists(directory):
        os.makedirs(directory)


random.seed(0)

def distributed_is_initialized():
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def beta_policy(mom_fn):
    def _alr(optimizer, iteration, epoch, param, indx):
        mom = mom_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group[param][indx] = mom

    return _alr


def mom_cosine_policy(base_beta, warmup_length, epochs):
    def _beta_fn(iteration, epoch):
        if epoch < warmup_length:
            beta = base_beta * (epoch + 1) / warmup_length
        else:
            beta = base_beta
        return beta

    return beta_policy(_beta_fn)


def clip(image_tensor, use_fp16=False):
    '''
    Adjust the input based on mean and variance.
    Automatically determines if the image is grayscale (1 channel) or RGB (3 channels).
    
    Parameters:
    - image_tensor: The input tensor to be clipped.
    - use_fp16: Whether to use half-precision floating-point numbers.
    
    Returns:
    - image_tensor: The adjusted tensor.
    '''
    # Determine if the image is RGB or grayscale based on the number of channels
    num_channels = image_tensor.shape[1]  # Assuming shape is [batch_size, channels, height, width]
    
    if num_channels == 3:
        # RGB settings (for ImageNet or similar datasets)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16 if use_fp16 else np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16 if use_fp16 else np.float32)
    elif num_channels == 1:
        # Grayscale settings (for MNIST)
        mean = np.array([0.1307], dtype=np.float16 if use_fp16 else np.float32)
        std = np.array([0.3081], dtype=np.float16 if use_fp16 else np.float32)
    else:
        raise ValueError("Unsupported number of channels. Expected 1 for grayscale or 3 for RGB images.")

    # Apply the clipping based on mean and std
    for c in range(num_channels):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)

    return image_tensor


def denormalize(image_tensor, use_fp16=False):
    '''
    convert floats back to input
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


def color_jitter(x, brightness=0.2, contrast=0.2):
    """
    Applies color jittering by adjusting brightness and contrast to a grayscale image.
    Args:
    - x (torch.Tensor): Input image tensor with shape (1, H, W) for MNIST.
    - brightness (float): Factor to adjust brightness. A higher value increases brightness.
    - contrast (float): Factor to adjust contrast. A higher value increases contrast.
    
    Returns:
    - torch.Tensor: Color jittered image.
    """
    # Brightness adjustment
    # Adding a random brightness factor
    brightness_factor = 1.0 + (torch.rand(1).item() * 2 - 1) * brightness  # Between 1-brightness and 1+brightness
    x_bright = x * brightness_factor
    
    # Contrast adjustment
    mean = x_bright.mean()
    contrast_factor = 1.0 + (torch.rand(1).item() * 2 - 1) * contrast  # Between 1-contrast and 1+contrast
    x_jittered = (x_bright - mean) * contrast_factor + mean

    return x_jittered

import torch.nn.functional as F
def differentiable_rotate_translate_blur(images, angle_range=5, translate_range=2, blur_sigma=1):
    """
    Apply differentiable rotation, translation, and Gaussian blur to a batch of images.
    
    Args:
    - images (torch.Tensor): Input image tensor of shape (B, C, H, W).
    - angle_range (float): Maximum angle for random rotation in degrees.
    - translate_range (float): Maximum translation in pixels for x and y directions.
    - blur_sigma (float): Standard deviation for Gaussian blur kernel.
    
    Returns:
    - torch.Tensor: Augmented images with the same shape as input.
    """
    device = images.device
    B, C, H, W = images.shape
    
    # Rotation
    angles = (torch.rand(B, device=device) * 2 - 1) * angle_range  # Random angles in range [-angle_range, angle_range]
    cos_vals = torch.cos(torch.deg2rad(angles))
    sin_vals = torch.sin(torch.deg2rad(angles))

    # Translation
    translate_x = (torch.rand(B, device=device) * 2 - 1) * translate_range
    translate_y = (torch.rand(B, device=device) * 2 - 1) * translate_range

    # Construct affine transformation matrices
    theta = torch.zeros((B, 2, 3), device=device)
    theta[:, 0, 0] = cos_vals
    theta[:, 0, 1] = -sin_vals
    theta[:, 1, 0] = sin_vals
    theta[:, 1, 1] = cos_vals
    theta[:, 0, 2] = translate_x / (W / 2)  # Normalize for affine grid
    theta[:, 1, 2] = translate_y / (H / 2)  # Normalize for affine grid

    # Apply affine transformation (rotation + translation)
    grid = F.affine_grid(theta, images.size(), align_corners=True)
    transformed_images = F.grid_sample(images, grid, align_corners=True)

    # Gaussian Blur
    kernel_size = int(2 * 3 * blur_sigma + 1)  # Calculate kernel size based on sigma
    x = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2
    gauss_kernel = torch.exp(-0.5 * (x / blur_sigma) ** 2)
    gauss_kernel = gauss_kernel / gauss_kernel.sum()

    # Create 2D Gaussian kernel from 1D kernels
    gauss_kernel_2d = gauss_kernel[:, None] @ gauss_kernel[None, :]
    gauss_kernel_2d = gauss_kernel_2d.expand(C, 1, kernel_size, kernel_size)  # Shape (C, 1, K, K) for depthwise conv

    # Apply Gaussian blur using depthwise convolution
    padding = kernel_size // 2
    blurred_images = F.conv2d(transformed_images, gauss_kernel_2d, padding=padding, groups=C)

    return blurred_images