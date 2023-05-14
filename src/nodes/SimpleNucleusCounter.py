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
        self.final_output = None

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
        anti_mask = ip.generate_anti_contour(self.contour_img).astype(np.uint8)
        self.prepared_img = ip.preprocess_img(raw_img).astype(np.uint8)
        self.prepared_img *= anti_mask
        self.prepared_img += self.contour_img
        self.prepared_img = cv.cvtColor(self.prepared_img, cv.COLOR_BGR2GRAY)

    def plot_output(self):
        plt.figure(figsize=(12,8))
        plt.axis('off')
        plt.imshow(self.final_output)
        plt.pause(0.01) 

    def process(self):
        # Contour Hierarchy
        outline_pad = 32
        
        padded_outlined_img = cv.copyMakeBorder(
           self.prepared_img,
           outline_pad, outline_pad, outline_pad, outline_pad,
           cv.BORDER_CONSTANT, value=0
        )
        padded_outlined_img = cv.copyMakeBorder(
           padded_outlined_img,
           outline_pad, outline_pad, outline_pad, outline_pad,
           cv.BORDER_CONSTANT, value=255
        )
        th, threshed = cv.threshold(
           padded_outlined_img, 0, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU
        )
        contours, hierarchy = cv.findContours(
           threshed, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1
        )
        particle_counts = {}
        hierarchy = hierarchy[0] # Removing Redundant Dimension
        curr_parent_id = -1
        for curr_node_id in range(len(hierarchy)):
           parent_node_id = hierarchy[curr_node_id][3]
           if parent_node_id == -1:
              continue
           elif parent_node_id == 0:
              curr_parent_id = curr_node_id
              particle_counts[curr_parent_id] = 0
           # else:
           #    if curr_parent_id in particle_counts.keys():
           #       particle_counts[curr_parent_id] += 1

        # Figure with particle_ids
        final_output = padded_outlined_img.copy()
        final_output = cv.cvtColor(final_output, cv.COLOR_GRAY2BGR)
        for c_id in particle_counts.keys():
           rect = cv.minAreaRect(contours[c_id])
           box = cv.boxPoints(rect)
           box = np.int0(box)
           final_output = cv.drawContours(final_output, [box], 0, (255, 0, 0), 10)

        for c_id in particle_counts.keys():
           rect = cv.minAreaRect(contours[c_id])
           box = cv.boxPoints(rect)
           box = np.int0(box)
           x1 = box[0, 0]
           y1 = box[0, 1]
           x2 = box[2, 0]
           y2 = box[2, 1]
           rect_center = (int((x1+x2)/2), int((y1+y2)/2))
           text = str(c_id)
           cv.putText(img=final_output,
                      text=text,
                      org=rect_center,
                      fontFace=cv.FONT_HERSHEY_TRIPLEX,
                      fontScale=0.85,
                      color=(0,255,0),
                      thickness=2)
        
        self.final_output = final_output
        # file_name = output_folder + "/" + img_name + "_labeled_contours.png"
        # fig, ax = plt.subplots(figsize=(16, 16))
        # ax.imshow(final_output, cmap="gray")
        # plt.axis("off")
        # 
