import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
from src.nodes.AbstractNode import AbstractNode
from nd2reader import ND2Reader
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Agnostic Device
device = "cuda" if torch.cuda.is_available() else "cpu"

class InvalidChannelError(Exception):
    def __init__(self):
        msg = "Input channel id is either larger than the channels present "
        msg += " or input channel is below 0"
        super().__init__(msg)

class SamNucleusSegmenter(AbstractNode):
    def __init__(self):
        super().__init__()
        self.output_name = "SamNucleusMask"

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
        return mask_img

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


        

    def process(self):
        raw_img = self.get_raw_nucleus_img()
        prepared_img = self.preprocess_img(raw_img)
        sam_checkpoint = "sam_model/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(prepared_img)
        mask_img = self.generate_mask_img(prepared_img, masks)
        return mask_img
        
        
