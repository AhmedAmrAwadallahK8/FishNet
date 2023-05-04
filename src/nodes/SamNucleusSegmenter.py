import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
from src.nodes.AbstractNode import AbstractNode
from src.FishNet import FishNet
from nd2reader import ND2Reader

# Agnostic Device
device = "cuda" if torch.cuda.is_available() else "cpu"

class SamNucleusSegmenter(AbstractNode):
    def __init__(self):
        super.__init__()
        self.output_name = "SamNucleusMask"

    def scale_and_clip_img(self, img):
        mean = img.mean()
        std = img.std()
        # img_clip = np.clip(img, mean-1.5*std, mean+3*std)
        img_clip = np.clip(img, mean-1*std, mean+4*std)
        #img_clip[img_clip == img_clip.max()] = 1
        #img_clip[img_clip == 0] = img_clip.max()
        img_clip_scale = rescale_img(img_clip)
        img_clip_scale_denoise = cv2.fastNlMeansDenoising(img_clip_scale)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_clip_scale_denoise = clahe.apply(img_clip_scale_denoise)
        return img_clip_scale_denoise

    def preprocess_img(self, img):
        img = scale_and_clip_img(img)
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

    def process(self):
        raw_img = self.
        prepared_img = self.preprocess_img(image)
        sam_checkpoint = "../../sam_model/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(prepared_img)
        
        
