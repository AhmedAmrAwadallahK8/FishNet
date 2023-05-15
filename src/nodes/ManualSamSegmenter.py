import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
import src.user_interaction as usr_int
from src.nodes.AbstractNode import AbstractNode
from nd2reader import ND2Reader
from segment_anything import sam_model_registry, SamPredictor


class ManualSamSegmenter(AbstractNode):
    def __init__(self):
        super().__init__(output_name="NucleusMask",
                         requirements=[],
                         user_can_retry=True,
                         node_title="Manual SAM Segmenter")
        self.device = "cude" if torch.cuda.is_available() else "cpu"
        self.sam_mask_generator = None
        self.sam = None
        self.sam_predictor = None
        self.input_boxes = [[]]

    def setup_sam(self):
        sam_checkpoint = "sam_model/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.sam_predictor = SamPredictor(self.sam)

    def process(self):
        pass

    def initialize_node(self):
        self.setup_sam()

    def plot_output(self):
        pass
