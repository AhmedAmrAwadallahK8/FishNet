import cv2
import numpy as np

def generate_mask_img(base_img, masks):
    """
    Takes in pure SAM masks and, using the base image as a shape context, 
    produces an intermediate mask image data type used for all future
    mask operations

    Args:
        base_img (ndarray): base image SAM was used on
        masks (list): masks returned by pure SAM

    Returns:
        ndarray: 2D numpy array where each mask is now represented by an ID
    """
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

def generate_mask_img_manual(base_img, masks):
    """
    Takes in manual SAM masks and, using the base image as a shape context, 
    produces an intermediate mask image data type used for all future
    mask operations

    Args:
        base_img (ndarray): base image SAM was used on
        masks (list): masks returned by manual SAM

    Returns:
        ndarray: 2D numpy array where each mask is now represented by an ID
    """
    mask_shape = (base_img.shape[0], base_img.shape[1])
    mask_img = np.zeros(mask_shape)
    instance_id = 0
    for m in masks:
        m = m[0,:,:]
        instance_id += 1
        mask_instance = np.zeros(mask_shape)
        segment_instance = np.where(m == True, instance_id, 0)
        mask_instance += segment_instance
        mask_img = np.where(
            (mask_img == 0) &
            (mask_instance > 0),
            mask_instance,
            mask_img)
        # mask_img += mask_instance
    return mask_img.astype(int)
