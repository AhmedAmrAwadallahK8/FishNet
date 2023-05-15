import cv2
import numpy as np

# This shouldn't be here probably
class InvalidChannelError(Exception):
    def __init__(self):
        msg = "Input channel id is either larger than the channels present "
        msg += " or input channel is below 0"
        super().__init__(msg)

# This shouldn't be here probably
def check_if_valid_channel(channel):
    from src.fishnet import FishNet
    channel_count = FishNet.raw_imgs.shape[1]
    if channel > (channel_count - 1) or channel < 0:
        raise InvalidChannelError
         
# This shouldn't be here probably
def get_raw_nucleus_img():
    from src.fishnet import FishNet
    nucleus_channel = int(input("Specify the Nucleus axis id: "))
    check_if_valid_channel(nucleus_channel)
    raw_img = FishNet.raw_imgs[0][nucleus_channel]
    return raw_img

def rescale_img(img):
    img_scaled = img.copy()
    if np.amax(img) > 255:
        img_scaled = cv2.convertScaleAbs(img, alpha = (255.0/np.amax(img)))
    else:
        img_scaled = cv2.convertScaleAbs(img)
    return img_scaled

def scale_and_clip_img(img):
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

def preprocess_img(img):
    img = scale_and_clip_img(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    return img


def generate_contour_img(mask_img):
    contour_col = (255, 255, 255)
    contour_shape = (mask_img.shape[0], mask_img.shape[1], 3)
    contour_img = np.zeros(contour_shape, dtype=np.uint8)
    gray_mask = mask_img.astype(np.uint8)
    # gray_mask = cv2.cvtColor(mask_img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    
    cnts = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(contour_img, [c], -1, contour_col, thickness=2)
    return contour_img

def generate_anti_contour(base_contour):
    anti_mask = np.where(base_contour > 0, 0, 1)
    return anti_mask

def generate_activation_mask(mask_img):
    act_mask = np.where(mask_img > 0, 1, 0).astype(np.uint8)
    act_mask = cv2.cvtColor(act_mask, cv2.COLOR_GRAY2BGR)
    return act_mask
    

def generate_colored_contour(mask_img, contour_color):
    contour_shape = (mask_img.shape[0], mask_img.shape[1], 3)
    contour_img = np.zeros(contour_shape, dtype=np.uint8)
    gray_mask = mask_img.astype(np.uint8)
    # gray_mask = cv2.cvtColor(mask_img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    
    cnts = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(contour_img, [c], -1, contour_col, thickness=2)
    return contour_img
