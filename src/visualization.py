import os
import cv2
import numpy as np
from .loader import Node, ImageContainer

def load_lut(filename):
    """Loads a LUT color map in RGB format."""

    colormap_dir = os.path.join(os.path.dirname(__file__), 'colormaps')
    colormap_path = os.path.join(colormap_dir, (filename + ".lut"))
    
    if not os.path.exists(colormap_path):
        colormap_files = os.listdir(colormap_dir)
        # Display list of available colormaps
        print(f"Color map '{filename}' not found in directory. Available color maps:")
        for file in colormap_files:
            print(os.path.splitext(file)[0])
        return None
    
    lut = np.loadtxt(colormap_path, dtype=np.uint8, delimiter=',', skiprows=1)
    return np.reshape(lut[:, 1:], (1, 256, 3))


def to_multichannel(img):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img_bgr.astype(np.uint8)


class ImageViewer(Node):
    def __init__(self, name="ImageViewer", colormap=None):
        super().__init__(name=name)
        self.valid_inputs = ImageContainer
        self.valid_outputs = ImageContainer
        self.inputs = ImageContainer
        self.outputs = None
        self.channel_index = 0
        self.zstack_index = 0
        self.colormap = colormap

    def process(self, inputs):
        image_container = inputs
        image = image_container.image
        print(image.shape)

        image = image[0, self.channel_index, self.zstack_index, :, :]

        if self.colormap:
            lut = load_lut(self.colormap)
            norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            multi = to_multichannel(norm)
            image = cv2.LUT(multi, cv2.cvtColor(lut, cv2.COLOR_BGR2RGB))
        
        cv2.imshow("Image Viewer", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return image_container
    
    def check_valid_inputs(self, inputs):
        print('check valid inputs in ImageViewer:')
        # ND2Loader node should have no inputs
        return isinstance(inputs, self.valid_inputs)

    
    def check_valid_outputs(self, outputs):
        print('check valid outputs')
        return isinstance(outputs, self.valid_outputs)



