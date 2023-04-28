import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from .common import Node
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def show_anns(anns):
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


class SegmentAnything(Node):

    def __init__(self, channel=0):
        self.channel = channel
        
    def process(self, image):

        print(image.size)

        # sam_checkpoint = "sam_vit_h_4b8939.pth"
        # model_type = "vit_h"

        # device = "cuda"

        # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        # sam.to(device=device)

        # mask_generator = SamAutomaticMaskGenerator(sam)

        # masks = mask_generator.generate(image)

        # print(len(masks))
        # print(masks[0].keys())
        # plt.figure(figsize=(20,20))
        # plt.imshow(image)
        # show_anns(masks)
        # plt.axis('off')
        # plt.show() 

        return image
