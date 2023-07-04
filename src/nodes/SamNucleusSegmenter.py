import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
import src.user_interaction as usr_int
from src.nodes.AbstractNode import AbstractNode
from nd2reader import ND2Reader
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

"""
This file is not used in FishNet
"""

class InvalidChannelError(Exception):
    def __init__(self):
        msg = "Input channel id is either larger than the channels present "
        msg += " or input channel is below 0"
        super().__init__(msg)

class SamNucleusSegmenter(AbstractNode):
    def __init__(self):
        super().__init__(output_name="NucleusMask",
                         requirements=[],
                         user_can_retry=True,
                         node_title="SAM Nucleus Segmenter")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam_mask_generator = None
        self.prepared_img = None
        self.mask_img = None
        self.sam = None
        self.save_file_name = "sam_nucleus_segmentation.png"
        self.default_sam_settings = {
            "points_per_side": 32, #32
            "pred_iou_thresh": 0.97, #0.95
            "stability_score_thresh": 0.97, #0.97
            "crop_n_layers": 1, #1
            "crop_n_points_downscale_factor": 2,
            "min_mask_region_area": 800 #200
        }
        self.custom_sam_settings = {
            "points_per_side": 0,
            "pred_iou_thresh": 0,
            "stability_score_thresh": 0,
            "crop_n_layers": 0,
            "crop_n_points_downscale_factor": 0,
            "min_mask_region_area": 0
        }
        self.sam_param_range = {
            "points_per_side": [4, 128],
            "pred_iou_thresh": [0, 1],
            "stability_score_thresh": [0, 1],
            "crop_n_layers": [0, 4],
            "crop_n_points_downscale_factor": [1, 8],
            "min_mask_region_area": [1, 10000]
        }
        self.sam_param_type = {
            "points_per_side": "int",
            "pred_iou_thresh": "float",
            "stability_score_thresh": "float",
            "crop_n_layers": "int",
            "crop_n_points_downscale_factor": "int",
            "min_mask_region_area": "int"
        }

        self.final_sam_settings = self.default_sam_settings.copy()

        pps_descrip = f"Points Per Side is the number of points to be sampled"
        pps_descrip += f" along one side of an image."
        pps_descrip += f" Larger values of points per side allows the model"
        pps_descrip += f" to segment more complex shapes."
        pps_descrip += f"\nValid values are within"
        pps_descrip += f" {self.sam_param_range['points_per_side']}"
        pps_descrip += f"\nDefault value is"
        pps_descrip += f" {self.default_sam_settings['points_per_side']}."

        iou_descrip = f"Predicted IOU Threshold uses an internal metric to "
        iou_descrip += f" assess the quality of a segmentation."
        iou_descrip += f" Lower values for this metric results in more"
        iou_descrip += f" segmentation instances of lower quality while"
        iou_descrip += f" higher is less instances of higher quality."
        iou_descrip += f"\nValid values are within"
        iou_descrip += f" {self.sam_param_range['pred_iou_thresh']}"
        iou_descrip += f"\nDefault value is"
        iou_descrip += f" {self.default_sam_settings['pred_iou_thresh']}."

        sst_descrip = f"Stability Score Threshold uses an internal metric"
        sst_descrip += f" that asses the stability of segmentation under"
        sst_descrip += f" changes to the cutoff. Higher values of this metric"
        sst_descrip += f" produces segmentations that rarely change while"
        sst_descrip += f" lower values produces output that often changes."
        sst_descrip += f"\nValid values for this metric are within"
        sst_descrip += f" {self.sam_param_range['stability_score_thresh']}"
        sst_descrip += f"\nDefault value is"
        sst_descrip += f" {self.default_sam_settings['stability_score_thresh']}."

        cnl_descrip = f"Crop N Layers causes the model to run predictions on"
        cnl_descrip += f" crops of the image a well. This is another method"
        cnl_descrip += f" of controlling for segmentation of stability but"
        cnl_descrip += f" it comes at a large computation cost."
        cnl_descrip += f"\nValid values for this metric are within"
        cnl_descrip += f" {self.sam_param_range['crop_n_layers']}"
        cnl_descrip += f"\nDefault value is"
        cnl_descrip += f" {self.default_sam_settings['crop_n_layers']}."

        cnpdf_descrip = "NA Description"
        cnpdf_descrip += f"\nValid values for this metric are within "
        cnpdf_descrip += f"{self.sam_param_range['crop_n_points_downscale_factor']}"
        cnpdf_descrip += f"\nDefault value is "
        cnpdf_descrip += f"{self.default_sam_settings['crop_n_points_downscale_factor']}."

        mmra_descrip = f"Min Mask Region Area is a post processing parameter"
        mmra_descrip += f" that removes segmentations that are below the"
        mmra_descrip += f" specified area. Size depends on the object you are"
        mmra_descrip += f" trying to segment."
        mmra_descrip += f"\nValid values for this metric are within"
        mmra_descrip += f" {self.sam_param_range['min_mask_region_area']}"
        mmra_descrip += f"\nDefault value is"
        mmra_descrip += f" {self.default_sam_settings['min_mask_region_area']}"


        self.param_description = {
            "points_per_side": pps_descrip, #What does this even mean
            "pred_iou_thresh": iou_descrip,
            "stability_score_thresh": sst_descrip,
            "crop_n_layers": cnl_descrip,
            "crop_n_points_downscale_factor": cnpdf_descrip,
            "min_mask_region_area": mmra_descrip
        }

    def rescale_img(self, img):
        img_scaled = img.copy()
        if np.amax(img) > 255:
            img_scaled = cv2.convertScaleAbs(img, alpha = (255.0/np.amax(img)))
        else:
            img_scaled = cv2.convertScaleAbs(img)
        return img_scaled

    def scale_and_clip_img(self, img):
        mean = img.mean()
        std = img.std()
        # img_clip = np.clip(img, mean-1.5*std, mean+3*std)
        img_clip = np.clip(img, mean-1*std, mean+4*std)
        #img_clip[img_clip == img_clip.max()] = 1
        #img_clip[img_clip == 0] = img_clip.max()
        img_clip_scale = self.rescale_img(img_clip)
        img_clip_scale_denoise = cv2.fastNlMeansDenoising(img_clip_scale)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_clip_scale_denoise = clahe.apply(img_clip_scale_denoise)
        return img_clip_scale_denoise

    def preprocess_img(self, img):
        img = self.scale_and_clip_img(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
        return img

    def show_anns(self, anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        for ann in sorted_anns:
            m = ann['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:,:,i] = color_mask[i]
            ax.imshow(np.dstack((img, m*0.35)))

    def generate_colored_mask(self, mask_img):
        color_shape = (mask_img.shape[0], mask_img.shape[1], 3)
        color_mask = np.zeros(color_shape)
        for segment_id in range(1, int(mask_img.max())+1):
            id_instance = np.where(mask_img == segment_id, segment_id, 0)
            color_instance = np.where(mask_img == segment_id, 1, 0)
            red_pix = np.random.randint(256, size=1)[0]
            green_pix = np.random.randint(256, size=1)[0]
            blue_pix = np.random.randint(256, size=1)[0]

            #When the color is close to black default to making it white
            if (red_pix + green_pix + blue_pix) <= 5:
                red_pix = 255
                green_pix = 255
                blue_pix = 255
            color_mask[:,:,0] += color_instance*red_pix
            color_mask[:,:,1] += color_instance*green_pix
            color_mask[:,:,2] += color_instance*blue_pix
        color_mask = color_mask.astype(int)
        return color_mask

    # In the future change so that the entire process is 2d
    def generate_mask_img(self, img, sam_masks):
        mask_img = np.zeros(img.shape)
        instance_id = 0
        for instance in sam_masks:
            instance_id += 1
            mask_instance = np.zeros(img.shape)
            segment_instance = np.where(instance["segmentation"] == True, instance_id, 0)
            mask_instance[:,:,0] =  (mask_instance[:,:,0] + segment_instance)
            mask_instance[:,:,1] =  (mask_instance[:,:,1] + segment_instance)
            mask_instance[:,:,2] =  (mask_instance[:,:,2] + segment_instance)
            mask_img = mask_img + mask_instance
        mask_img = mask_img.astype(int)
        return mask_img[:,:,0]

    def generate_contour_img(self, mask_img):
        contour_col = (255, 0, 0)
        contour_shape = (mask_img.shape[0], mask_img.shape[1], 3)
        contour_img = np.ones(contour_shape, dtype=np.uint8) * 255
        gray_mask = mask_img.astype(np.uint8)
        # gray_mask = cv2.cvtColor(mask_img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        
        cnts = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(contour_img, [c], -1, contour_col, thickness=2)
        return contour_img

    def check_if_valid_channel(self, channel):
        from src.fishnet import FishNet
        channel_count = FishNet.raw_imgs.shape[1]
        if channel > (channel_count - 1) or channel < 0:
            raise InvalidChannelError
             

    def get_raw_nucleus_img(self):
        from src.fishnet import FishNet
        nucleus_channel = int(input("Specify the Nucleus axis id: "))
        self.check_if_valid_channel(nucleus_channel)
        raw_img = FishNet.raw_imgs[0][nucleus_channel]
        return raw_img

    def setup_sam(self):
        print("Setting up SAM model...")
        sam_checkpoint = "sam_model/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)
        self.sam_mask_generator = SamAutomaticMaskGenerator(
                                      model=self.sam,
                                      **self.final_sam_settings)
        

    def get_sam_segment_data(self, prepared_img):
        print("Segmenting Nucleus this step takes some time...")
        sam_masks = self.sam_mask_generator.generate(prepared_img)
        return sam_masks

    def plot_output(self):
        #Plot for user to examine...
        colored_mask = self.generate_colored_mask(self.mask_img)
        contour_img = self.generate_contour_img(self.mask_img)
        output_compare = np.hstack((self.prepared_img, colored_mask, contour_img))
        plt.figure(figsize=(12,8))
        plt.axis('off')
        plt.imshow(output_compare)
        plt.pause(0.01)

    def get_custom_sam_parameters(self):
        print("")
        for param in self.custom_sam_settings:
            param_message = self.param_description[param]
            print(param_message)
            msg = f"Input a value for {param}: "
            param_range = self.sam_param_range[param]
            param_setting = usr_int.get_numeric_input_in_range(msg, param_range)
            if self.sam_param_type[param] == "int":
                param_setting = int(param_setting)
            self.custom_sam_settings[param] = param_setting
            print("")
            

    # I dont like this name
    def custom_or_default(self):
        prompt = "Would you like to use custom settings for SAM? "
        response_id = usr_int.ask_user_for_yes_or_no(prompt)
        if response_id == usr_int.positive_response_id:
            self.get_custom_sam_parameters()
            self.final_sam_settings = self.custom_sam_settings.copy()
        elif response_id == usr_int.negative_response_id:
            print("Using default SAM parameters...")
            self.final_sam_settings = self.default_sam_settings.copy()

    def prepare_sam_hyperparameters(self):
        msg = "The SAM Nucleus Segmenter can use either default or custom "
        msg += "parameters."
        print(msg)
        self.custom_or_default()

    def initialize_node(self):
        raw_img = self.get_raw_nucleus_img()
        self.prepared_img = self.preprocess_img(raw_img)
        self.prepare_sam_hyperparameters()
        self.setup_sam()

    def quick_sam_setup(self):
        self.sam_mask_generator = SamAutomaticMaskGenerator(
                                      model=self.sam,
                                      **self.final_sam_settings)


    def reinitialize_node(self):
        self.prepare_sam_hyperparameters()
        self.quick_sam_setup()

    def save_output(self):
        self.save_img(self.mask_img, self.save_file_name)
        # folder_name = FishNet.save_folder
        # mask_file_path = folder_name + self.save_file_name
        # fig, ax = plt.subplots(figsize=(16, 16))
        # plt.axis("off")
        # ax.imshow(self.mask_img, cmap="gray")
        # plt.savefig(mask_file_path, bbox_inches="tight")
        # plt.close(fig)

    def process(self):
        sam_masks = self.get_sam_segment_data(self.prepared_img)
        self.mask_img = self.generate_mask_img(self.prepared_img, sam_masks)
        return self.mask_img


