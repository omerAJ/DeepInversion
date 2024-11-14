# --------------------------------------------------------
# Modified from NVIDIA Corporation's Dreaming to Distill for MNIST
# --------------------------------------------------------

from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import argparse
import torch
from torch import distributed, nn
import random
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import torch.cuda.amp as amp
import os
import torchvision.models as models
from utils.utils import load_model_pytorch, distributed_is_initialized, color_jitter

random.seed(0)

import torch.nn as nn
import torchvision.models as models

class MobileNetV2Mnist(nn.Module):
    def __init__(self):
        super(MobileNetV2Mnist, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        
        # Modify the first layer to accept 1 channel (grayscale)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Modify the classifier to output 10 classes (for MNIST)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 10)
        
    def forward(self, x):
        return self.model(x)


# Define the ResNet50 adaptation for MNIST
class ResNet50Mnist(nn.Module):
    def __init__(self):
        super(ResNet50Mnist, self).__init__()
        self.model = models.resnet50(pretrained=True)
        # Modify the input layer to accept 1 channel (grayscale)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        # Modify the final layer to output 10 classes (for MNIST)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)
        
    def forward(self, x):
        return self.model(x)

def validate_one(input, target, model):
    """Perform validation on the validation set where target is a probability distribution or a one-hot label."""
    
    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k, adapted for both softmax and one-hot target."""
        maxk = max(topk)
        batch_size = target.size(0)

        if target.dim() > 1 and target.size(1) > 1:
            _, true_class = target.max(dim=1)
        else:
            true_class = target

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(true_class.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        output = model(input)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    print("Verifier accuracy: ", prec1.item())

def run(args):
    torch.manual_seed(args.local_rank)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    # Load the custom ResNet50 for MNIST
    if args.arch_name == "resnet50mnist":
        print("Loading ResNet50 adapted for MNIST")
        net = ResNet50Mnist()
    else:
        raise ValueError("Unsupported architecture for this script")

    net = net.to(device)

    print('==> Resuming from checkpoint..')

    # Load the MNIST model
    path_to_model = "D:\omer\DeepInversion\models\only5SamplesZeroClass_resnet50_mnist.pth"
    def load_model_pytorch(model, checkpoint_path, gpu_n=None):
        """
        Load model weights from a PyTorch checkpoint file.
        
        Parameters:
        - model: The model instance to load weights into.
        - checkpoint_path: Path to the checkpoint file (e.g., .pth or .pt).
        - gpu_n: Specifies the GPU device index if using a GPU. Defaults to None.
        
        Returns:
        - model: Model with loaded weights.
        """
        # Load the checkpoint data
        if gpu_n is not None:
            checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{gpu_n}')
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check if the checkpoint contains 'state_dict' (common in training checkpoints)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove module prefix if saved with DataParallel or DistributedDataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # Remove "module." prefix if present
            new_state_dict[name] = v
        
        # Load the state_dict into the model
        model.load_state_dict(new_state_dict)
        
        print(f"Model loaded from {checkpoint_path}")
        return model
    load_model_pytorch(net, path_to_model, gpu_n=torch.cuda.current_device())

    net.to(device)
    net.eval()

    # Load verifier model
    net_verifier = None
    if args.verifier and args.adi_scale == 0:
        if args.local_rank == 0:
            print("loading verifier: ", args.verifier_arch)
            net_verifier = models.__dict__[args.verifier_arch](pretrained=True).to(device)
            net_verifier.eval()

    # Configuration parameters
    from deepinversion import DeepInversionClass

    exp_name = args.exp_name
    adi_data_path = "./final_images/%s" % exp_name
    exp_name = "generations/%s" % exp_name

    args.iterations = 2000
    args.start_noise = True

    args.resolution = 28  # Set resolution for MNIST images
    bs = args.bs
    jitter = 0  # No jitter for MNIST

    parameters = dict()
    parameters["resolution"] = 28
    parameters["random_label"] = False
    parameters["start_noise"] = True
    parameters["detach_student"] = False
    parameters["do_flip"] = False  # No flip for MNIST

    parameters["do_flip"] = args.do_flip
    parameters["random_label"] = args.random_label
    parameters["store_best_images"] = args.store_best_images

    criterion = nn.CrossEntropyLoss()

    coefficients = dict()
    coefficients["r_feature"] = args.r_feature
    coefficients["first_bn_multiplier"] = args.first_bn_multiplier
    coefficients["tv_l1"] = args.tv_l1
    coefficients["tv_l2"] = args.tv_l2
    coefficients["l2"] = args.l2
    coefficients["lr"] = args.lr
    coefficients["main_loss_multiplier"] = args.main_loss_multiplier
    coefficients["adi_scale"] = args.adi_scale

    network_output_function = lambda x: x

    # check accuracy of verifier
    if args.verifier:
        hook_for_display = lambda x, y: validate_one(x, y, net_verifier)
    else:
        hook_for_display = None

    DeepInversionEngine = DeepInversionClass(
        net_teacher=net,
        final_data_path=adi_data_path,
        path=exp_name,
        parameters=parameters,
        setting_id=args.setting_id,
        bs=bs,
        use_fp16=args.fp16,
        jitter=jitter,
        criterion=criterion,
        coefficients=coefficients,
        network_output_function=network_output_function,
        hook_for_display=hook_for_display,
        dataset="mnist"
    )

    net_student = None
    if args.adi_scale != 0:
        net_student = net_verifier

    DeepInversionEngine.generate_batch(net_student=net_student)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worldsize', type=int, default=1, help='Number of processes participating in the job.')
    parser.add_argument('--local_rank', type=int, default=0, help='Rank of the current process.')
    parser.add_argument('--adi_scale', type=float, default=0.0, help='Coefficient for Adaptive Deep Inversion')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--epochs', default=20000, type=int, help='number of epochs')
    parser.add_argument('--setting_id', default=0, type=int, help='settings for optimization')
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--arch_name', default='resnet50mnist', type=str, help='model name for MNIST')

    parser.add_argument('--fp16', action='store_true', help='use FP16 for optimization')
    parser.add_argument('--exp_name', type=str, default='temp', help='where to store experimental data')

    parser.add_argument('--verifier', action='store_true', help='evaluate batch with another model')
    parser.add_argument('--verifier_arch', type=str, default='mobilenet_v2', help="verifier arch name")

    parser.add_argument('--do_flip', action='store_true', help='apply flip during model inversion')
    parser.add_argument('--random_label', action='store_true', help='generate random label for optimization')
    parser.add_argument('--r_feature', type=float, default=0.05, help='feature distribution regularization')
    parser.add_argument('--first_bn_multiplier', type=float, default=10.0, help='first BN layer multiplier')
    parser.add_argument('--tv_l1', type=float, default=0.0, help='TV L1 loss coefficient')
    parser.add_argument('--tv_l2', type=float, default=0.0001, help='TV L2 loss coefficient')
    parser.add_argument('--lr', type=float, default=0.2, help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, help='L2 loss coefficient')
    parser.add_argument('--main_loss_multiplier', type=float, default=1.0, help='main loss coefficient')
    parser.add_argument('--store_best_images', action='store_true', help='save best images')

    args = parser.parse_args()
    print(args)

    torch.backends.cudnn.benchmark = True
    run(args)

if __name__ == '__main__':
    main()
