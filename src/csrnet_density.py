# src/csrnet_density.py
"""
CSRNet Density Estimation

Generates density maps using:
 - CSRNet (for dense crowds)
 - Gaussian fallback (for testing or missing model)
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
# from models.CSRNet_pytorch.model import CSRNet as OriginalCSRNet

class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()

        # ----- Front end config ----- #
        # these numbers follow a VGG-like structure
        # each int = number of filters in a conv layer
        # 'M' = max pooling layer
        frontend_feat = [64,64,'M',128,128,'M',256,256,256,'M',512,512,512]

        # ----- Back end config ----- #
        # After frontend, CSRNet adds dilated convolutions to increase receptive field without losing resolution
        backend_feat = [512,512,512,256,128,64]

        # build frontend CNN using sequential conv + relu (+ maxpool)
        self.frontend = self._make_layers(frontend_feat)
        # build backend CNN with dilation (larger field of view)
        self.backend = self._make_layers(backend_feat, in_channels=512, dilation=True)
        # final 1×1 conv to output density map with a single channel
        self.output_layer = nn.Conv2d(64,1,kernel_size=1)

    def forward(self, x):
        # pass input through frontend layers
        x = self.frontend(x)

        # pass through dilated backend layers
        x = self.backend(x)

        # 1×1 conv outputs density map
        x = self.output_layer(x)

        return x


    def _make_layers(self, cfg, in_channels=3, dilation=False):
        """
        Dynamically build CNN layers based on a config list (cfg):
        - If item is an integer: Conv2d + ReLU
        - If item is 'M': MaxPool2d
        """
        layers = []
        # set dilation rate for backend
        d = 2 if dilation else 1
        for v in cfg:
            if v == 'M':
                # downsample by factor 2
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
            else:
                # 3×3 convolution with optional dilation and padding
                conv2d = nn.Conv2d(in_channels, v, 3, padding=d, dilation=d)
                layers += [conv2d, nn.ReLU(inplace=True)]
                # update number of channels for next layer
                in_channels = v

        # unpack - take the list [Conv2d, ReLU, MaxPool2d, Conv2d, ReLU] -> expand to Conv2d, ReLU, MaxPool2d, Conv2d, ReLU
        return nn.Sequential(*layers)

class DensityEstimator:
    def __init__(self, weights_path=None):
        # use GPU if available, else CPU - to make it compatible with windows systems
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize CSRNet model
        self.model = CSRNet().to(self.device)
        # self.model = OriginalCSRNet().to(self.device)

        # image preprocessing:
        # convert to tensor
        # norm using ImageNet mean/std (standard practice)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

        # load weights if provided
        if weights_path:
            # load checkpoint file from disk
            ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
            # some checkpoints wrap weights in 'state_dict'
            sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            # load weights into model
            self.model.load_state_dict(sd)
            # Remove "module." prefix if present (DataParallel checkpoint)
            # clean_sd = {}
            # for k, v in sd.items():
            #     new_key = k.replace("module.", "")  # remove DP prefix
            #     clean_sd[new_key] = v
            #
            # self.model.load_state_dict(clean_sd)
            # set model to eval mode
            self.model.eval()
            print("CSRNet loaded from:", weights_path)
        else:
            # if no pretrained weights: model will produce random output
            print("CSRNet weights not found. Using fake density.")

    def get_density_map(self, frame):
        """
        Runs CSRNet on an image (frame) and returns the predicted density map.
        """
        # convert BGR frame → RGB PIL Image (PyTorch models use RGB)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # apply preprocessing transform and add batch dimension (1, C, H, W)
        x = self.transform(img).unsqueeze(0).to(self.device)
        # disable gradient computation for inference
        with torch.no_grad():
            # model output: (1 × 1 × H × W)
            den = self.model(x).cpu().numpy().squeeze()

        # return the density map as a 2D numpy array
        return den

    def fake_density(self, frame):
        """
        Fallback density map if CSRNet weights are missing.
        Creates a smoothed grayscale heatmap as a simple approximation.
        """
        # convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # apply a large Gaussian blur to mimic smooth density variation
        heatmap = cv2.GaussianBlur(gray, (51,51), 0)
        # norm values to range [0, 1]
        return cv2.normalize(heatmap.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)