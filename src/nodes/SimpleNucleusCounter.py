import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import src.user_interaction as usr_int
import src.image_processing as ip
from src.nodes.AbstractNode import AbstractNode





class SimpleNucleusCounter(AbstractNode):
    def __init__(self):
        self.nucleus_mask_req = "NucleusMask"
        super().__init__(output_name="SimpleNucleusCount",
                         requirements=[self.nucleus_mask_req],
                         user_can_retry=True,
                         node_title="Simple Nucleus Counter")
        self.nucleus_mask = None
        self.contour_img = None
        self.prepared_img = None

    def generate_contour_img(self, mask_img):
        contour_col = (255, 0, 0)
        contour_shape = (mask_img.shape[0], mask_img.shape[1], 3)
        contour_img = np.ones(contour_shape, dtype=np.uint8)*255
        gray_mask = mask_img.astype(np.uint8)

        cnts = cv.findContours(gray_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv.drawContours(contour_img, [c], -1, contour_col, thickness=1)
        return contour_img

    def initialize_node(self):
        from src.fishnet import FishNet
        self.nucleus_mask = FishNet.pipeline_output[self.nucleus_mask_req]
        self.contour_img = ip.generate_contour_img(self.nucleus_mask)
        raw_img = ip.get_raw_nucleus_img()
        anti_mask = np.where(self.contour_img > 0, 0, 1).astype(np.uint8)
        self.prepared_img = ip.preprocess_img(raw_img).astype(np.uint8)
        self.prepared_img *= anti_mask
        self.prepared_img += self.contour_img
        pass

    def plot_output(self):
        plt.figure(figsize=(12,8))
        plt.axis('off')
        plt.imshow(self.prepared_img)
        plt.pause(0.01) 

    def process(self):
        
        pass
        
