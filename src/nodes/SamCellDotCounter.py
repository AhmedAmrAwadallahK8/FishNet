import numpy as np
import random
import torch
import torchvision
import cv2
import cv2 as cv
from src.nodes.AbstractNode import AbstractNode
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import src.image_processing as ip
import src.sam_processing as sp
import src.user_interaction as usr_int
import os
import time

class SamCellDotCounter(AbstractNode):
    def __init__(self):
        from src.fishnet import FishNet
        super().__init__(output_name="SamDotCountPack",
                         requirements=["ManualCellMaskPack"],
                         user_can_retry=False,
                         node_title="Auto SAM Cell Dot Counter")
        self.skip_node = True
        self.save_folder = "particle_segmentations/"
        self.max_pix_area = 1024*1024 #1024*1024
        self.quilt_factor = 1
        self.base_img = None
        self.cyto_id_mask = None
        self.nuc_id_mask = None
        self.cytoplasm_key = "cyto"
        self.nucleus_key = "nuc"
        self.cell_id_mask = None
        self.csv_name =  "dot_counts.csv"
        self.nuc_counts = {}
        self.cyto_counts = {}
        self.seg_imgs = {}
        self.raw_crop_imgs = {}
        save_folder = FishNet.save_folder + self.save_folder
        self.process_times = []
        self.total_cell_count = 0
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        self.csv_data = []

        self.z_key = "Z Axis"
        self.c_key = "Channel Axis"
        self.quilt_key = "Quilt Factor"
        self.z = None
        self.c = None

        self.settings = [
            self.quilt_key,
            self.z_key,
            self.c_key]
        self.user_settings = {}
        self.setting_type = {
            self.z_key: "categ",
            self.c_key: "categ",
            self.quilt_key: "int"}
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
                user_setting = usr_int.get_categorical_input_set_in_range(
                    msg,
                    setting_range)
            elif self.setting_type[setting] == "int":
                user_setting = usr_int.get_numeric_input_in_range(
                    msg,
                    setting_range)
                user_setting = int(user_setting)
            self.user_settings[setting] = user_setting
            print("")

    def finish_setting_setup(self):
        from src.fishnet import FishNet
        self.setting_range = {
            self.quilt_key: [1,4],
            self.z_key: list(FishNet.z_meta.keys()),
            self.c_key: list(FishNet.channel_meta.keys())}
        z_descrip = f"Enter all the z axes that you are interested in "
        z_descrip += f"seperated by commas\n"
        z_descrip += f"Valid z axes are in {self.setting_range[self.z_key]}."

        c_descrip = f"Enter all the channels that you are interested in "
        c_descrip += f"seperated by commas\n"
        c_descrip += f"Valid channels are in {self.setting_range[self.c_key]}."

        quilt_descrip = f"The quilt factor increase performance at segmenting"
        quilt_descrip += f" small objects at the cost of increased "
        quilt_descrip += f"computation time.\n"
        quilt_descrip += f"Recommended value is 2.\n"
        quilt_descrip += f"Valid channels are in "
        quilt_descrip += f"{self.setting_range[self.quilt_key]}."
        self.setting_description = {
            self.z_key: z_descrip,
            self.c_key: c_descrip,
            self.quilt_key: quilt_descrip}

    def update_context_img(self):
        from src.fishnet import FishNet
        c_ind = FishNet.channel_meta[self.c]
        z_ind = FishNet.z_meta[self.z]
        raw_img = ip.get_specified_channel_combo_img([c_ind], [z_ind])
        self.base_img = ip.preprocess_img2(raw_img)

    def setup_sam(self):
        from src.fishnet import FishNet
        default_sam_settings = {
                    "points_per_side": 32, #32
                    "pred_iou_thresh": 0.5, #0.5
                    "stability_score_thresh": 0.90, #0.95
                    "crop_n_layers": 0, #1
                    "crop_n_points_downscale_factor": 0, #2
                    "min_mask_region_area": 10 } #1
        FishNet.sam_model.setup_auto_mask_pred(default_sam_settings)


    def initialize_node(self):
        prompt = "The Sam Cell Dot Counter Node can take a long time to process"
        prompt += " especially without gpu support.\n"
        if torch.cuda.is_available():
            prompt += "Currently you have gpu support.\n"
        else:
            prompt += "Currently you do not have gpu support.\n"
        prompt += "Enter yes if you would like to continue, no to quit: "
        usr_response_id = usr_int.ask_user_for_yes_or_no(prompt)
        if usr_response_id == usr_int.positive_response_id:
            self.skip_node = False
        else:
            return
        self.finish_setting_setup()
        self.get_id_mask()
        self.setup_sam()
        self.get_setting_selections_from_user()
        self.quilt_factor = self.user_settings[self.quilt_key]
        self.total_cell_count = np.max(self.cyto_id_mask)

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
        self.save_dot_count_csv()
        self.save_segs()

    def process(self):
        if self.skip_node:
            self.set_node_as_successful()
            return
        print(f"Percent Done: 0.00%, Estimated Time Completion To: NA")
        process_count = 0
        total_count = len(self.user_settings[self.z_key])
        total_count *= len(self.user_settings[self.c_key])
        for z_axis in self.user_settings[self.z_key]:
            for c_axis in self.user_settings[self.c_key]:
                process_count += 1
                self.z = z_axis
                self.c = c_axis
                self.update_context_img()

                start_time = time.time()
                self.process_cell_part(self.cytoplasm_key)
                self.process_cell_part(self.nucleus_key)
                end_time = time.time()
                elapsed_time = end_time - start_time

                self.process_times.append(elapsed_time)
                self.store_csv_data()

                mean_time = np.mean(self.process_times)
                steps_to_go = total_count-process_count
                time_left = steps_to_go*mean_time/60
                percent_done = process_count/total_count*100
                prog_msg = f"Percent Done: {percent_done:.2f}%, Estimated"
                prog_msg += f"Time To Completion: {time_left:.2f}min"
                print(prog_msg)
        self.set_node_as_successful()

    def store_csv_data(self):
        for cell_id in self.nuc_counts.keys():
            if cell_id in self.cyto_counts.keys():
                obs = f"{cell_id},{self.cyto_counts[cell_id]:d},"
                obs += f"{self.nuc_counts[cell_id]:d},{self.z},{self.c}\n"
                self.csv_data.append(obs)

    def save_dot_count_csv(self):
        from src.fishnet import FishNet
        # csv of particle counts
        particle_csv = FishNet.save_folder + self.csv_name
        csv_file = open(particle_csv, "w")
        csv_file.write("cell_id,cyto_counts,nuc_counts,z_level,channel\n")
        for obs in self.csv_data:
                csv_file.write(obs)
        csv_file.write("\n")
        csv_file.close()

    def save_segs(self):
        for save_name in self.seg_imgs:
            img_path = self.save_folder + save_name
            self.save_img(self.seg_imgs[save_name], img_path)
        for save_name in self.raw_crop_imgs:
            img_path = self.save_folder + save_name
            self.save_img(self.raw_crop_imgs[save_name], img_path)

    def process_cell_part(self, cell_part):
        id_mask = self.cell_id_mask[cell_part]
        cell_ids = np.unique(id_mask)
        for cell_id in cell_ids:
            if cell_id == 0 or cell_id > self.max_cell_id:
                continue

            targ_shape = self.base_img.shape
            id_activation = np.where(id_mask == cell_id, 1, 0)
            resized_id_activation = id_activation[:, :, np.newaxis]
            resized_id_activation = ip.resize_img(
                resized_id_activation,
                targ_shape[0],
                targ_shape[1],
                inter_type="linear")
            resized_id_activation = resized_id_activation[:, :, np.newaxis]
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
            img_crop = img_id_activated[ymin:ymax, xmin:xmax, :].copy()
            img_crop_h = img_crop.shape[0]
            img_crop_w = img_crop.shape[1]
            scale_factor = np.sqrt(self.max_pix_area/(img_crop_h*img_crop_w))
            img_crop = ip.resize_img(
                img_crop,
                int(img_crop_h*scale_factor),
                int(img_crop_w*scale_factor))

            dot_count = None
            seg = None
            if self.quilt_factor < 2:
                dot_count, seg = self.get_dot_count_and_seg_pure(img_crop.copy())
            elif self.quilt_factor >= 2:
                dot_count, seg = self.get_dot_count_and_seg_quilt(img_crop.copy())
            if cell_part == self.cytoplasm_key:
                self.cyto_counts[cell_id] = dot_count
            elif cell_part == self.nucleus_key:
                self.nuc_counts[cell_id] = dot_count
            self.store_segmentation(cell_part, cell_id, img_crop, seg)
            if cell_part == self.cytoplasm_key:
                self.store_raw_crop(id_bbox, cell_id)

    def store_raw_crop(self, id_bbox, cell_id):
        base_shape = self.base_img.shape
        pad = 20
        xmin = int(id_bbox[0] - pad)
        xmax = int(id_bbox[2] + pad)
        ymin = int(id_bbox[1] - pad)
        ymax = int(id_bbox[3] + pad)
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax >= base_shape[1]:
            xmax = base_shape[1]-1
        if ymax >= base_shape[0]:
            ymax = base_shape[0] - 1
        save_name = f"cell{cell_id}_z{self.z}_{self.c}_raw.png"
        self.raw_crop_imgs[save_name] = self.base_img[ymin:ymax, xmin:xmax, :].copy()

    def store_segmentation(self, cell_part, cell_id, orig_img, segmentation):
        img_overlay = np.where(segmentation > 0, segmentation, orig_img)
        save_name = f"cell{cell_id}_{cell_part}_z{self.z}_{self.c}_seg.png"
        self.seg_imgs[save_name] = img_overlay

    def get_segmentation_bbox(self, single_id_mask):
        gray = single_id_mask[:, :, np.newaxis].astype(np.uint8)
        contours, hierarchy = cv2.findContours(
            gray,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE)[-2:]
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

    def get_dot_count_and_seg_quilt(self, img_subset):
        img_seq = self.get_image_seq(img_subset)
        seg_seq, dot_count = self.get_dot_count_and_seg_seq(img_seq)
        restored_seg = self.coalesce_img_seq(img_subset, seg_seq)
        return dot_count, restored_seg

    def get_dot_count_and_seg_pure(self, img_subset):
        from src.fishnet import FishNet
        mask = FishNet.sam_model.get_auto_mask_pred(img_subset)
        mask_img, dot_count = self.process_sam_mask(img_subset, mask)
        seg = ip.generate_single_colored_mask(mask_img)
        return dot_count, seg

    def coalesce_img_seq(self, img, img_seq):
        y_block_size = int(img.shape[1]/self.quilt_factor)
        x_block_size = int(img.shape[0]/self.quilt_factor)
        i = 0
        for x_img in range(self.quilt_factor):
            for y_img in range(self.quilt_factor):
                start_x = x_img*x_block_size
                start_y = y_img*y_block_size
                end_x = x_img*x_block_size + x_block_size
                end_y = y_img*y_block_size + y_block_size
                img[start_x:end_x, start_y:end_y, :] = img_seq[i].astype(int)
                i += 1
        return img

    def get_image_seq(self, img):
        img_seq = []
        y_block_size = int(img.shape[1]/self.quilt_factor)
        x_block_size = int(img.shape[0]/self.quilt_factor)
        for x_img in range(self.quilt_factor):
            for y_img in range(self.quilt_factor):
                start_x = x_img*x_block_size
                start_y = y_img*y_block_size
                end_x = x_img*x_block_size + x_block_size
                end_y = y_img*y_block_size + y_block_size
                img_seq.append(img[start_x:end_x, start_y:end_y])
        return img_seq

    def get_dot_count_and_seg_seq(self, img_seq):
        from src.fishnet import FishNet
        seg_seq = []
        total_dot_count = 0

        for img in img_seq:
            masks = FishNet.sam_model.get_auto_mask_pred(img)
            mask_img, dot_counts = self.process_sam_mask(img, masks)
            total_dot_count += dot_counts
            seg = ip.generate_single_colored_mask(mask_img)
            seg_seq.append(seg)
        return seg_seq, total_dot_count

    def process_sam_mask(self, img, sam_mask):
        mask_shape = (img.shape[0], img.shape[1])
        mask_img = np.zeros(mask_shape)
        total_pix = np.sum(np.ones(mask_shape))
        instance_id = 0
        for m in sam_mask:
                mask_sum = np.sum(m["segmentation"])
                if mask_sum/total_pix > 0.05:
                    continue
                instance_id += 1
                mask_instance = np.zeros(mask_shape)
                segment_instance = np.where(
                    m["segmentation"] == True,
                    instance_id,
                    0)
                mask_instance += segment_instance
                mask_img += mask_instance
        return mask_img, instance_id
