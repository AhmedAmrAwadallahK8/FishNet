import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
from nd2reader import ND2Reader

# Agnostic Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Sa
