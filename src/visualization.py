import cv2
import numpy as np
from .loader import Node, ImageContainer

def load_lut(filename):
    """Loads a LUT color map in RGB format."""
    lut = np.loadtxt(filename, dtype=np.uint8, delimiter=',', skiprows=1)
    return np.reshape(lut[:, 1:], (1, 256, 3))


def to_multichannel(img):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img_bgr.astype(np.uint8)


class ImageViewer(Node):
    def __init__(self, name="ImageViewer"):
        super().__init__(name=name)
        self.valid_inputs = ImageContainer
        self.valid_outputs = ImageContainer
        self.inputs = ImageContainer
        self.outputs = None
        self.channel_index = 0
        self.zstack_index = 0

    def process(self, inputs):
        image_container = inputs
        image = image_container.image
        print(image.shape)
        image = image[0, self.channel_index, self.zstack_index, :, :]
        cv2.imshow("Image Viewer", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return [image_container]
    
    def check_valid_inputs(self, inputs):
        print('check valid inputs in ImageViewer:')
        # ND2Loader node should have no inputs
        return isinstance(inputs, self.valid_inputs)

    
    def check_valid_outputs(self, outputs):
        print('check valid outputs')
        return isinstance(outputs, self.valid_outputs)



