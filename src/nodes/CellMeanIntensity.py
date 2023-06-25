import numpy as np
import cv2
import cv2 as cv
from src.nodes.AbstractNode import AbstractNode
import src.image_processing as ip
import src.user_interaction as usr_int
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
        self.csv_name =  "cell_intensity.csv"
        self.nuc_intensity = {}
        self.cyto_intensity = {}
        self.nuc_area = {}
        self.cyto_area = {}
        self.nuc_intensity_sum = {}
        self.cyto_intensity_sum = {}
        self.channel_context = {}
        self.z_context = {}
        self.raw_crop_imgs = {}
        self.max_cyto_id = 0
        self.csv_data = []

        self.z_key = "Z Axis"
        self.c_key = "Channel Axis"
        self.z = None
        self.c = None

        self.settings = [
            self.z_key,
            self.c_key]
        self.user_settings = {}
        self.setting_type = {
            self.z_key: "categ",
            self.c_key: "categ"}
        self.setting_range = {}
        self.setting_description = {}


    def get_setting_selections_from_user(self):
        print("")
        for setting in self.settings:
            user_setting = None
            setting_message = self.setting_description[setting]
            print(setting_message)
            msg = f"Input a value for {setting}: "
            setting_range = self.setting_range[setting]
            if self.setting_type[setting] == "categ":
                user_setting = usr_int.get_categorical_input_set_in_range(msg, setting_range)
            self.user_settings[setting] = user_setting
            print("")

    def finish_setting_setup(self):
        from src.fishnet import FishNet
        self.setting_range = {
            self.z_key: list(FishNet.z_meta.keys()),
            self.c_key: list(FishNet.channel_meta.keys())}
        z_descrip = f"Enter all the z axi that you are interested in seperated by commas\n"
        z_descrip += f"Valid z axi are in {self.setting_range[self.z_key]}."

        c_descrip = f"Enter all the channels that you are interested in seperated by commas\n"
        c_descrip += f"Valid channels are in {self.setting_range[self.c_key]}."
        self.setting_description = {
            self.z_key: z_descrip,
            self.c_key: c_descrip}
        

    def initialize_node(self):
        # raw_img = ip.get_all_mrna_img()
        # self.base_img = raw_img.copy()
        self.finish_setting_setup()
        self.get_id_mask()
        self.get_setting_selections_from_user()

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

    def update_context_img(self):
        from src.fishnet import FishNet
        c_ind = FishNet.channel_meta[self.c]
        z_ind = FishNet.z_meta[self.z]
        self.base_img = ip.get_specified_channel_combo_img([c_ind], [z_ind])

    def process(self):
        print(f"Percent Done: 0.00%")
        process_count = 0 
        total_count = len(self.user_settings[self.z_key])
        total_count *= len(self.user_settings[self.c_key])
        for z_axis in self.user_settings[self.z_key]:
            for c_axis in self.user_settings[self.c_key]:
                process_count += 1
                self.z = z_axis
                self.c = c_axis
                self.update_context_img()
                self.process_cell_part(self.cytoplasm_key)
                self.process_cell_part(self.nucleus_key)
                self.store_csv_data()
                percent_done = process_count/total_count*100
                print(f"Percent Done: {percent_done:.2f}%")
        self.set_node_as_successful()

    def store_csv_data(self):
        for cell_id in self.nuc_intensity_sum.keys():
            # if cell_id in self.cyto_intensity:
                # obs = f"{cell_id},{self.cyto_intensity[cell_id]:.3f},{self.nuc_intensity[cell_id]:.3f},{self.z},{self.c}\n"
                obs = f"{cell_id},{self.cyto_intensity_sum[cell_id]:.1f},{self.cyto_area[cell_id]:.1f},{self.nuc_intensity_sum[cell_id]:.1f},{self.nuc_area[cell_id]:.1f},{self.z},{sel.c}\n"
                self.csv_data.append(obs)
        
    

    def save_csv(self):
        from src.fishnet import FishNet
        # csv of particle counts
        csv_path = FishNet.save_folder + self.csv_name
        csv_file = open(csv_path, "w")
        csv_file.write("cell_id,cyto_intensity_sum,cyto_area,nuc_intensity_sum,nuc_area,z_level,channel\n")
        for obs in self.csv_data:
                csv_file.write(obs)
        csv_file.write("\n")
        csv_file.close()

    def process_cell_part(self, cell_part):
        # print(f"Processing {cell_part}...")
        id_mask = self.cell_id_mask[cell_part]
        cell_ids = np.unique(id_mask)
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
            # mean_intensity = self.calc_mean_intensity(img_crop)
            area, intensity_sum = self.get_area_and_intensity_sum(img_crop)
            if cell_part == self.cytoplasm_key:
                # self.cyto_intensity[cell_id] = mean_intensity
                self.cyto_area[cell_id] = area
                self.cyto_intensity_sum[cell_id] = intensity_sum
            elif cell_part == self.nucleus_key:
                # self.nuc_intensity[cell_id] = mean_intensity
                self.nuc_area[cell_id] = area
                self.nuc_intensity_sum[cell_id] = intensity_sum

    def get_area_and_intensity_sum(self, img_crop):
        area = np.sum(np.where(img_crop > 0, 1, 0))
        intensity_sum = np.sum(img_crop)
        return area, intensity_sum

    def calc_mean_intensity(self, img_crop):
        total_pix = np.sum(np.where(img_crop > 0, 1, 0))
        intensity_sum = np.sum(img_crop)
        mean_intensity = intensity_sum/total_pix
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
