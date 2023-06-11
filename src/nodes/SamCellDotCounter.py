# Counts dots present in nucleus and cell
# Save an image showing what the model counted so researcher can finish

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
        self.base_img = None
        self.sam_mask_generator = None
        self.sam = None
        self.sam_predictor = None
        self.cyto_id_mask = Fishnet.pipeline_output["cytoplasm"]
        self.nuc_id_mask = Fishnet.pipeline_output["nucleus"]
        self.nuc_counts = {}
        self.cyto_counts = {}

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
        process_cytos()
        process_nucs()
        self.set_node_as_successful()
        pass

    def process_cytos(self):
        cyto_ids = np.unique(self.cyto_id_mask)
        for cyto_id in cyto_ids:
            if cyto_id == 0:
                continue
            id_activation = np.where(self.cyto_id_mask == cyto_id, 1, 0)
            id_bbox = self.get_segmentation_bbox(id_activation)
            img_id_activated = id_activation * self.base_img
            img_crop = img_id_activated[id_bbox]
            self.cyto_counts[cyto_id] = self.get_dot_count(img_crop)

    def process_nucs(self):
        nuc_ids = np.unique(self.nuc_id_mask)
        for nuc_id in nuc_ids:
            if nuc_id == 0:
                continue
            id_activation = np.where(self.nuc_id_mask == nuc_id, 1, 0)
            id_bbox = self.get_segmentation_bbox(id_activation)
            img_id_activated = id_activation * self.base_img
            img_crop = img_id_activated[id_bbox]
            self.nuc_counts[nuc_id] = self.get_dot_count(img_crop)

    def get_segmentation_bbox(self, single_id_mask):
        gray = single_id_mask[:, :, np.newaxis].astype(np.uint8)
        contours, hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        idx =0 
        rois = []
        largest_area = 0
        best_bbox = []
        first = True
        for cnt in contours:
            idx += 1
            area = cv.contourArea(cnt)
            rect_pack = cv2.boundingRect(cnt) #x, y, w, h
            x, y, w, h = rect_pack
            bbox = [x, y, x+w, y+h]
            if first:
                first = False
                largest_area = area
                best_bbox = bbox
            else:
                if area > largest_area:
                    largest_area = area
                    best_bbox = bbox
        return best_bbox

    def get_dot_count(self, img_subset):
        pass


