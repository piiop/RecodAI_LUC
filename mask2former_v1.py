import os
import cv2
import json
import sympy
import sympy.printing  # ensure submodule is imported
sympy.printing = sympy.printing  # attach as attribute explicitly
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.ops import FeaturePyramidNetwork

from scipy.optimize import linear_sum_assignment

import albumentations as A
import matplotlib.pyplot as plt


from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import MaskRCNN
from sklearn.model_selection import train_test_split
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F_transforms