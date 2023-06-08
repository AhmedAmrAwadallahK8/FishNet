# Counts dots present in nucleus and cell

import numpy as np
import torch
import torchvision
import cv2
from src.nodes.AbstractNode import AbstractNode
from segment_anything import sam_model_registry, SamPredictor
import src.image_processing as ip
import src.sam_processing as sp

class SamCellDotCounter(AbstractNode):
    def __init__(self):
        from src.fishnet import Fishnet
        super().__init__(output_name="SamDotCountPack",
                         requirements=["ManualCellMaskPack"],
                         user_can_retry=False,
                         node_title="Auto SAM Cell Dot Counter")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam_mask_generator = None
        self.sam = None
        self.sam_predictor = None
        self.cyto_id_mask = Fishnet.pipeline_output["cytoplasm"]
        self.nuc_id_mask = Fishnet.pipeline_output["nucleus"]

    def setup_sam(self):
        sam_checkpoint = "sam_model/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)
        self.sam_predictor = SamPredictor(self.sam)
        # self.sam_predictor.set_image(img)


    def initialize_node(self):
        # Image Prep?
        self.setup_sam()

    def save_output(self):
        pass

    def process(self):
        pass
        
        


