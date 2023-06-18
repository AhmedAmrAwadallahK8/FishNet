import numpy as np
import cv2
import cv2 as cv
from src.nodes.AbstractNode import AbstractNode
import src.image_processing as ip
import os

class CellMeanIntensity(AbstractNode):
    def __init__(self):
        from src.fishnet import FishNet
        super().__init__(output_name="CellMeanIntensityPack",
                         requirements=["ManualCellMaskPack"],
                         user_can_retry=False,
                         node_title="Cell Mean Intensity")
        self.base_img = None
        self.cyto_id_mask = None
        self.nuc_id_mask = None
        self.cytoplasm_key = "cyto"
        self.nucleus_key = "nuc"
        self.cell_id_mask = None
        self.csv_name =  "mean_intensity.csv"
        self.nuc_intensity = {}
        self.cyto_intensity = {}
        self.raw_crop_imgs = {}
        self.max_cyto_id = 0

    def initialize_node(self):
        raw_img = ip.get_all_mrna_img()
        self.base_img = raw_img.copy()
        self.get_id_mask()

    def get_id_mask(self):
        from src.fishnet import FishNet
        mask_pack = FishNet.pipeline_output["ManualCellMaskPack"]
        self.cyto_id_mask = mask_pack["cytoplasm"]
        self.nuc_id_mask = mask_pack["nucleus"]
        self.cell_id_mask = {
            self.cytoplasm_key: self.cyto_id_mask,
            self.nucleus_key: self.nuc_id_mask
        }
        self.max_cell_id = np.max(self.cyto_id_mask)
        

    def save_output(self):
        self.save_csv()

    def process(self):
        self.process_cell_part(self.cytoplasm_key)
        self.process_cell_part(self.nucleus_key)
        self.set_node_as_successful()

    def save_csv(self):
        from src.fishnet import FishNet
        # csv of particle counts
        csv_path = FishNet.save_folder + self.csv_name
        csv_file = open(csv_path, "w")
        csv_file.write("cell_id,cyto_mean_intensity,nuc_mean_intensity\n")
        for cell_id in self.nuc_intensity.keys():
            if cell_id in self.cyto_intensity:
                obs = f"{cell_id},{self.cyto_intensity[cell_id]:.3f},{self.nuc_intensity[cell_id]:.3f}\n"
                csv_file.write(obs)
        csv_file.write("\n")
        csv_file.close()

    def process_cell_part(self, cell_part):
        print(f"Processing {cell_part}...")
        id_mask = self.cell_id_mask[cell_part]
        cell_ids = np.unique(id_mask)
        print(f"Percent Done: 0.00%")
        for cell_id in cell_ids:
            if cell_id == 0 or cell_id > self.max_cell_id:
                continue

            targ_shape = self.base_img.shape
            id_activation = np.where(id_mask == cell_id, 1, 0)
            resized_id_activation = ip.resize_img(
                id_activation,
                targ_shape[0],
                targ_shape[1],
                inter_type="linear")
            id_bbox = self.get_segmentation_bbox(id_activation)
            id_bbox = ip.rescale_boxes(
                [id_bbox],
                id_activation.shape,
                self.base_img.shape)[0]
            xmin = int(id_bbox[0])
            xmax = int(id_bbox[2])
            ymin = int(id_bbox[1])
            ymax = int(id_bbox[3])
            img_id_activated = resized_id_activation * self.base_img
            img_crop = img_id_activated[ymin:ymax, xmin:xmax].copy()
            mean_intensity = self.calc_mean_intensity(img_crop)
            if cell_part == self.cytoplasm_key:
                self.cyto_intensity[cell_id] = mean_intensity
            elif cell_part == self.nucleus_key:
                self.nuc_intensity[cell_id] = mean_intensity
            percent_done = cell_id / (len(cell_ids)-1)*100
            print(f"Overall percent Done: {percent_done:.2f}%")

    def calc_mean_intensity(self, img_crop):
        total_pix = np.sum(np.where(img_crop > 0, 1, 0))
        mean_intensity = np.sum(img_crop)/total_pix
        return mean_intensity

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
