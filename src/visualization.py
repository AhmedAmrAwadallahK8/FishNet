import cv2
import numpy as np

def load_lut(filename):
    """Loads a LUT color map in RGB format."""
    lut = np.loadtxt(filename, dtype=np.uint8, delimiter=',', skiprows=1)
    return np.reshape(lut[:, 1:], (1, 256, 3))


def to_multichannel(img):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img_bgr.astype(np.uint8)