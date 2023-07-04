import cv2 as cv

supported_img_files = ["jpg", "png"]

def process_img_type(img, ext):
    """
    Converts supported imgs into the expected format for this program which is
    a HxWxC where the C axis is of length 3

    Args:
        img (ndarray): numpy array containing data related to an image
        ext (str): extension associated with the loaded image

    Returns:
        ndarray: numpy array containing the image in expected format
    """
    if ext == "jpg":
        return img
    elif ext == "png":
        img = img[:,:,:3]
        return img

def load_img_file():
    """
    Prompts user for image file, extracts the image, processes it, and then
    returns it

    Args:
        Nothing

    Returns:
        ndarray: numpy array containing the image in expected format
    """
    img_path = input("Specify Img File Path: ")
    img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
    img_extension = img_path.split(".")[-1]
    img = process_img_type(img, img_extension)
    return img
