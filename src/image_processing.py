import cv2
import numpy as np

"""
Contains various functions associated with processing image data
"""

class InvalidChannelError(Exception):
    """
    Error associated to when a selected channel is not within the boundaries
    of the nd2 file
    """
    def __init__(self):
        msg = "Input channel id is either larger than the channels present "
        msg += " or input channel is below 0"
        super().__init__(msg)

def check_if_valid_channel(channel):
    """
    Checks if the input channel is within the bounds of the nd2 file within
    FishNet. If not raises the InvalidChannelError

    Args:
        Nothing

    Returns:
        Nothing
    """
    from src.fishnet import FishNet
    channel_count = FishNet.raw_imgs.shape[1]
    if channel > (channel_count - 1) or channel < 0:
        raise InvalidChannelError
         
def get_raw_nucleus_img():
    """
    User inputs the channel associated with a type of Nucleus staining. This
    image is then loaded and returned.

    Args:
        Nothing

    Returns:
        ndarray: numpy array associated with a Nucleus image
    """
    from src.fishnet import FishNet
    nucleus_channel = int(input("Specify the Nucleus axis id: "))
    check_if_valid_channel(nucleus_channel)
    raw_img = FishNet.raw_imgs[0][nucleus_channel]
    return raw_img

def get_all_channel_img():
    """
    Function purely designed for testing convenience and is not generalizable.
    It creates a simple linear combination of all the unique channels at 
    z axis 0 

    Args:
        Nothing

    Returns:
        ndarray: linear combination of all channels in our testing image
    """
    from src.fishnet import FishNet
    raw_img = FishNet.raw_imgs[0][0].copy()
    raw_img += FishNet.raw_imgs[0][1].copy()
    raw_img += FishNet.raw_imgs[0][2].copy()
    raw_img += FishNet.raw_imgs[0][3].copy()
    return raw_img

def get_zerod_img():
    """
    Function purely designed for testing convenience and is not generalizable.
    It creates a 0 matrix the same shape as the first image in the local nd2
    stack.

    Args:
        Nothing

    Returns:
        ndarray: numpy matrix of zeroes same size as one of the nd2 images

    """
    from src.fishnet import FishNet
    zerod_img = np.zeros(FishNet.raw_imgs[0][0].shape)
    return zerod_img

# This function may have a generalizability issue
def get_specified_channel_combo_img(channels, z_axi):
    """
    Takes a list of desired channels and z axes and produces an image that is a
    linear combination of them all. If one of the arguments is empty a linear
    combination cannot be made so a zeroed matrix in the same size as an
    a normal image is returned instead.

    Args:
        channels (list): all the channels of interest
        z_axis (list): all the z axes of interest

    Returns:
        ndarray: numpy array containing image data

    """
    from src.fishnet import FishNet
    raw_img = None
    if len(channels) <= 0 or len(z_axi) <= 0:
        raw_img = np.zeros(FishNet.raw_imgs[0][0].shape)
        return raw_img

    first = True
    for z in z_axi:
        for c in channels:
            if first:
                first = False
                raw_img = FishNet.raw_imgs[z][c].copy()
            raw_img += FishNet.raw_imgs[z][c].copy()
    return raw_img
    

def get_all_mrna_img():
    """
    Function purely designed for testing convenience and is not generalizable.
    Specifically selects the channel associated with a mRNA staining

    Args:
        Nothing

    Returns:
        ndarray: numpy array containing image data

    """
    from src.fishnet import FishNet
    raw_img = FishNet.raw_imgs[0][3]
    return raw_img

def rescale_img(img):
    """
    Function purely designed for testing convenience and is not generalizable.
    Specifically selects the channel associated with a mRNA staining

    Args:
        Nothing

    Returns:
        ndarray: numpy array containing image data

    """
    img_scaled = img.copy()
    if np.amax(img) > 255:
        img_scaled = cv2.convertScaleAbs(img, alpha = (255.0/np.amax(img)))
    else:
        img_scaled = cv2.convertScaleAbs(img)
    return img_scaled

def scale_and_clip_img(img):
    """
    Takes an image and performs max/min clipping, 8bit scaling, noise removal,
    and contrast enhancement.

    Args:
        img (ndarray): numpy array containing image data

    Returns:
        ndarray: numpy array containing image data

    """
    mean = img.mean()
    std = img.std()
    img_clip = np.clip(img, mean-1*std, mean+4*std)
    img_clip_scale = rescale_img(img_clip)
    img_clip_scale_denoise = cv2.fastNlMeansDenoising(img_clip_scale)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clip_scale_denoise = clahe.apply(img_clip_scale_denoise)
    return img_clip_scale_denoise

def preprocess_img2(img):
    """
    Takes a gray image and applies scale_and_clip_img. After it converts the
    image into a color image

    Args:
        img (ndarray): numpy array containing image data

    Returns:
        ndarray: numpy array containing image data

    """
    img = scale_and_clip_img(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def preprocess_img(img):
    """
    Takes a gray image and applies scale_and_clip_img. After it converts the
    image into a color image. Lastly it resizes the image to be 512x512

    Args:
        img (ndarray): numpy array containing image data

    Returns:
        ndarray: numpy array containing image data

    """
    img = scale_and_clip_img(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    return img

def resize_img_to_pixel_size2(img, targ_pixel_area):
    """
    Resizes an image to maintain the same HxW ratio but at a new pixel area

    Args:
        img (ndarray): numpy array containing image data
        targ_pixel_area (int): area of the target image size

    Returns:
        ndarray: numpy array containing image data

    """
    img_shape = img.shape
    pixel_area = img_shape[0]*img_shape[1]
    orig_h = img_shape[0]
    orig_w = img_shape[1]

    resize_factor = int(np.sqrt(targ_pixel_area/pixel_area))
    
    orig_h = int(orig_h)
    orig_w = int(orig_w)
    resized_h = int(orig_h*resize_factor)
    resized_w = int(orig_w*resize_factor)
    
    final_img = resize_img(img, resized_h, resized_w)
    
    return final_img

def resize_img_to_pixel_size(img, targ_pixel_area):
    """
    Resizes an image to maintain the same HxW ratio but at a new pixel area.
    Only proven for downsizing animage

    Args:
        img (ndarray): numpy array containing image data
        targ_pixel_area (int): area of the target image size

    Returns:
        ndarray: numpy array containing image data

    """
    img_shape = img.shape
    pixel_area = img_shape[0]*img_shape[1]
    
    orig_h = img_shape[0]
    orig_w = img_shape[1]
    hw_ratio = orig_h/orig_w
    new_w = 0
    new_h = 0
    
    if targ_pixel_area < pixel_area:
        x = orig_w - np.sqrt(orig_w*targ_pixel_area/orig_h)
        new_w = orig_w - x
        new_h = orig_h - x*hw_ratio
    elif targ_pixel_area > pixel_area: # Not 100% proven yet
        x = orig_w - np.sqrt(orig_w*targ_pixel_area/orig_h)
        new_w = orig_w + x
        new_h = orig_h + x*hw_ratio
    else:
        new_h = orig_h
        new_w = orig_w
        
    
    orig_h = int(orig_h)
    orig_w = int(orig_w)
    scaled_h = int(new_h)
    scaled_w = int(new_w)
    
    final_img = resize_img(img, scaled_h, scaled_w)
    
    return final_img

def rescale_boxes(boxes, orig_shape, targ_shape):
    """
    Resizes boxes draw in one image for a different sized version of the same
    image

    Args:
        boxes (ndarray): list of bbox coordinates
        orig_shape (int): area of the original image
        targ_shape (int): area of the target image
        

    Returns:
        list: list of boxes resized for the target image

    """
    orig_h = orig_shape[0]
    orig_w = orig_shape[1]
    scaled_h = targ_shape[0]
    scaled_w = targ_shape[1]
    boxes_np = np.asarray(boxes)
    orig_scale_coefs = np.asarray([
        orig_w, orig_h,
        orig_w, orig_h
    ])
    new_scale_coefs = np.asarray([
        scaled_w, scaled_h,
        scaled_w, scaled_h
    ])
    rescaled_boxes = boxes_np/orig_scale_coefs*new_scale_coefs
    return rescaled_boxes.tolist()
    

def resize_img(img, h, w, inter_type="cubic"):
    """
    Resizes an image based on input using opencv

    Args:
        img (ndarray): numpy array containing image data
        h (int): height of image
        w (int): width of image
        inter_type (str): specification for how to interpolate image resize

    Returns:
        ndarray: numpy array containing image data

    """
    img = img.astype(np.uint8)
    if inter_type == "cubic":
        img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    elif inter_type == "linear":
        img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
    else:
        print("INVALID INTERPOLATION TYPE, USING DEFALT INTERPOLATION TYPE")
        img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    return img

def generate_contour_img(mask_img):
    """
    Contours a mask img, already assumes a mask image is a gray image

    Args:
        mask_img (ndarray): numpy array containing mask data

    Returns:
        ndarray: contoured array

    """
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

def generate_dot_contour_img(mask_img):
    """
    Contours a mask img, already assumes a mask image is a gray image, 
    specialized for dots.

    Args:
        mask_img (ndarray): numpy array containing mask data

    Returns:
        ndarray: contoured array

    """
    contour_col = (0, 0, 255)
    contour_shape = (mask_img.shape[0], mask_img.shape[1], 3)
    contour_img = np.zeros(contour_shape, dtype=np.uint8)
    gray_mask = cv2.cvtColor(mask_img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    
    cnts = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(contour_img, [c], -1, contour_col, thickness=2)
    return contour_img

def generate_advanced_contour_img(mask_img):
    """
    Contours a mask img, already assumes a mask image is a gray image.
    Contouring is done independently for each unique id present within the mask.
    Takes longer to compute but has the benefit of having an independent
    contour for each isolated mask.

    Args:
        mask_img (ndarray): numpy array containing mask data

    Returns:
        ndarray: contoured array

    """
    contour_col = (255, 255, 255)
    contour_shape = (mask_img.shape[0], mask_img.shape[1], 3)
    final_contour_img = np.zeros(contour_shape, dtype=np.uint8)
    gray_mask = mask_img.astype(np.uint8)
    max_id = mask_img.max()

    # gray_mask = cv2.cvtColor(mask_img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    for id in range(1, max_id+1):
        contour_img = np.zeros(contour_shape, dtype=np.uint8)
        mask_instance = np.where(gray_mask == id, 255, 0).astype(np.uint8)
        cnts = cv2.findContours(mask_instance, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(contour_img, [c], -1, contour_col, thickness=2)
        final_contour_img += contour_img
        final_contour_img = np.where(final_contour_img > 0, 255, 0).astype(np.uint8)
    return final_contour_img

def add_label_to_img(img, mask_img):
    """
    Using a contour a label is placed in the center of the bounding box for the
    contour. This process is done by id instead of all at once

    Args:
        img (ndarray): original img
        mask_img (ndarray): numpy array containing mask data

    Returns:
        ndarray: image with labels where masks occur

    """
    # Prep
    max_id = mask_img.max()
    gray_mask = mask_img.astype(np.uint8)
    scale_factor = np.sqrt(img.shape[0]*img.shape[1]/(mask_img.shape[0]*mask_img.shape[1]))

    # Loop for every id
    for id in range(1, max_id+1):
        # Produce Contour
        mask_instance = np.where(gray_mask == id, 255, 0).astype(np.uint8)
        cnts = cv2.findContours(mask_instance, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        targ_contour = None
        largest_area = 0
        best_bbox = 0
        first = True

        # Only want the contour with the largest area
        for c in cnts:
            area = cv2.contourArea(c)
            rect_pack = cv2.boundingRect(c) #x, y, w, h
            x, y, w, h = rect_pack
            bbox = [x, y, x+w, y+h]
            if first:
                first = False
                targ_contour = c
                largest_area = area
                best_bbox = bbox
            else:
                if area > largest_area:
                    targ_contour = c
                    largest_area = area
                    best_bbox = bbox

        # Using contour place text
        rect = cv2.minAreaRect(targ_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        x1 = box[0, 0]*scale_factor
        y1 = box[0, 1]*scale_factor
        x2 = box[2, 0]*scale_factor
        y2 = box[2, 1]*scale_factor
        rect_center = (int((x1+x2)/2), int((y1+y2)/2))
        text = str(id)
        cv2.putText(
            img=img,
            text=text,
            org=rect_center,
            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
            fontScale=2,
            color=(0,255,0),
            thickness=2)
    return img


def generate_advanced_contour_with_label(mask_img):
    """
    Produce contour and label at the same time

    Args:
        mask_img (ndarray): numpy array containing mask data

    Returns:
        ndarray: blank image with labels and contours where masks occur

    """
    contour_col = (255, 255, 255)
    contour_shape = (mask_img.shape[0], mask_img.shape[1], 3)
    final_contour_img = np.zeros(contour_shape, dtype=np.uint8)
    gray_mask = mask_img.astype(np.uint8)
    max_id = mask_img.max()

    # gray_mask = cv2.cvtColor(mask_img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    for id in range(1, max_id+1):
        contour_img = np.zeros(contour_shape, dtype=np.uint8)
        mask_instance = np.where(gray_mask == id, 255, 0).astype(np.uint8)
        cnts = cv2.findContours(mask_instance, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(contour_img, [c], -1, contour_col, thickness=2)
            contour_img = np.where(contour_img > 0, 255, 0).astype(np.uint8)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            x1 = box[0, 0]
            y1 = box[0, 1]
            x2 = box[2, 0]
            y2 = box[2, 1]
            rect_center = (int((x1+x2)/2), int((y1+y2)/2))
            text = str(id)
            cv2.putText(
                img=contour_img,
                text=text,
                org=rect_center,
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=0.85,
                color=(0,255,0),
                thickness=2)
        final_contour_img += contour_img
        final_contour_img = np.where(final_contour_img > 255, 255, 0).astype(np.uint8)
        
    return final_contour_img

def generate_anti_contour(base_contour):
    """
    Takes a contour and makes everything else equal to 1 and the contour equal
    to 0

    Args:
        base_contour (ndarray): contour image

    Returns:
        ndarray: anti contour image

    """
    anti_mask = np.where(base_contour > 0, 0, 1)
    return anti_mask

def generate_activation_mask(mask_img):
    """
    Creates an activation mask. An activation mask is a mask where any value
    above 0 is 1 while 0 remains 0

    Args:
        mask_img (ndarray): numpy array containing mask data

    Returns:
        ndarray: activation mask

    """
    act_mask = np.where(mask_img > 0, 1, 0).astype(np.uint8)
    act_mask = cv2.cvtColor(act_mask, cv2.COLOR_GRAY2BGR)
    return act_mask
    

def generate_colored_contour(mask_img, contour_color):
    """
    Creates a contour of the mask_img with the desired color

    Args:
        mask_img (ndarray): numpy array containing mask data
        contour_color (tuple): RGB color 8bit

    Returns:
        ndarray: contour image

    """
    contour_shape = (mask_img.shape[0], mask_img.shape[1], 3)
    contour_img = np.zeros(contour_shape, dtype=np.uint8)
    gray_mask = mask_img.astype(np.uint8)
    # gray_mask = cv2.cvtColor(mask_img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    
    cnts = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(contour_img, [c], -1, contour_col, thickness=2)
    return contour_img

def generate_colored_mask(mask_img):
    """
    Creates a segmentation mask filled with random colors

    Args:
        mask_img (ndarray): numpy array containing mask data

    Returns:
        ndarray: segmentation mask image

    """
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

def generate_single_colored_mask(mask_img, color=(255, 0, 0)):
    """
    Creates a segmentation mask filled with a single color

    Args:
        mask_img (ndarray): numpy array containing mask data
        color (tuple): RGB color, red by default

    Returns:
        ndarray: segmentation mask image

    """
    color_shape = (mask_img.shape[0], mask_img.shape[1], 3)
    color_mask = np.zeros(color_shape)
    color_instance = np.where(mask_img > 0, 1, 0)
    red_pix = color[0]
    green_pix = color[1]
    blue_pix = color[2]
    # OPENCV uses BGR as the color channel order
    color_mask[:,:,2] += color_instance*red_pix
    color_mask[:,:,1] += color_instance*green_pix
    color_mask[:,:,0] += color_instance*blue_pix
    color_mask = color_mask.astype(int)
    return color_mask
