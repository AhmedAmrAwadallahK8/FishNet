import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import urllib.request
# from .common import Node 
from .loader import Node, ImageContainer
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
    def __init__(self, name="SegmentAnything", channel_index=0, zstack_index=0):
        super().__init__(name=name)
        self.valid_inputs = ImageContainer
        self.valid_outputs = ImageContainer
        self.inputs = ImageContainer
        self.outputs = None
        self.channel_index = channel_index
        self.zstack_index = zstack_index

    def process(self, inputs):
        image_container = inputs
        image = image_container.image

        image = image[0, self.channel_index, self.zstack_index, :, :]
        
        model_path = self.load_model()
        # sam_checkpoint = "models/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"

        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device=device)

        mask_generator = SamAutomaticMaskGenerator(sam)

        masks = mask_generator.generate(image)

        print(len(masks))
        print(masks[0].keys())
        plt.figure(figsize=(20,20))
        plt.imshow(image)
        show_anns(masks)
        plt.axis('off')
        plt.show() 
        
        return image_container
    
    # TODO: test the download model function
    def load_model(self):
        model_name = 'sam_vit_h_4b8939.pth'
        url = f'https://dl.fbaipublicfiles.com/segment_anything/{model_name}'
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        model_path = os.path.join(model_dir, model_name)
        print(model_path)
        if not os.path.exists(model_path):
            try:
                print('downloading model...')
                # urllib.request.urlretrieve(url)
                # print(f"Downloaded {filename} from {url}")
                return model_path
            except urllib.error.HTTPError:
                print('Error: file not found at URL')
        else:
            print(f"Model already exists.")
            return model_path
    
    def check_valid_inputs(self, inputs):
        print('check valid inputs in SegmentAnything:')
        # ND2Loader node should have no inputs
        return isinstance(inputs, self.valid_inputs)
    
    def check_valid_outputs(self, outputs):
        print('check valid outputs in SegmentAnything')
        # return isinstance(outputs, self.valid_outputs)
        return True
