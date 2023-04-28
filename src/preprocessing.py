import cv2 
import numpy as np 
from .common import Node 

class SelectChannel(Node):
    def __init__(self, channel):
        self.channel = channel
        
    def process(self, image):
        # Normalize image
        image.image = cv2.normalize(image.image, None, 0, 255, cv2.NORM_MINMAX)
        
        return image  

class ImageNormalization(Node):
    """
    A node that applies linear contrast enhancement to a grayscale image.
    
    Parameters
    ----------
    alpha : float, optional
        The scaling factor for the contrast enhancement. Values greater than 1.0
        increase the contrast of the image, while values less than 1.0 decrease
        the contrast. The default value is 1.0.
    beta : float, optional
        The shift factor for the contrast enhancement. This value is added to
        all pixel intensities in the image. Positive values increase the
        brightness of the image, while negative values decrease the brightness.
        The default value is 0.0.
    """

    def __init__(self):
        pass
        
    def process(self, image):
        # Normalize image
        image.image = cv2.normalize(image.image, None, 0, 255, cv2.NORM_MINMAX)
        
        return image        
    

class LinearContrast(Node):
    """
    A node that applies linear contrast enhancement to a grayscale image.
    
    Parameters
    ----------
    alpha : float, optional
        The scaling factor for the contrast enhancement. Values greater than 1.0
        increase the contrast of the image, while values less than 1.0 decrease
        the contrast. The default value is 1.0.
    beta : float, optional
        The shift factor for the contrast enhancement. This value is added to
        all pixel intensities in the image. Positive values increase the
        brightness of the image, while negative values decrease the brightness.
        The default value is 0.0.
    """

    def __init__(self, alpha=3.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        
    def process(self, image):

        # Apply contrast adjustment
        image.image = np.clip(self.alpha * image.image + self.beta, 0, 255).astype(np.uint8)
        
        return image




class CLAHEContrast(Node):
    """
    A node that applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to a grayscale image.
    
    Parameters
    ----------
    clip_limit : float, optional
        The contrast clip limit to use for CLAHE. This parameter controls the
        amount of contrast enhancement to apply. Higher values result in more
        enhancement, but can also introduce noise and artifacts. The default
        value is 2.0.
    tile_size : tuple of int, optional
        The size of the tiles to use for CLAHE. This parameter controls the
        size of the regions over which the histogram equalization is performed.
        Larger values result in more global contrast enhancement, but can also
        introduce artifacts. The default value is (8, 8).
    """

    def __init__(self, clip_limit=2.0, tile_size=8):
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        
    def process(self, image):        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.tile_size, self.tile_size))
        image.image = clahe.apply(image.image)
        
        return image