import cv2
import numpy as np

def generate_mask_img(base_img, masks):
    print(masks)
    mask_shape = (base_img.shape[0], base_img.shape[1])
    mask_img = np.zeros(mask_shape)
    instance_id = 0
    for m in masks:
            instance_id += 1
            mask_instance = np.zeros(mask_shape)
            segment_instance = np.where(m["segmentation"] == True, instance_id, 0)
            mask_instance += segment_instance
            mask_img += mask_instance
    return mask_img.astype(int)
