import os
import cv2
import nd2
import numpy as np 
from .common import Node

VALID_EXTENSIONS = [".tif", ".nd2"]


class ImageContainer(Node):
    def __init__(self, filename):
        self.filename = filename
        self.image = self.load()
        self.size = None
        self.dim = None
        self.n_channels = None
        self.n_zstack = None 

    # def get_size(self, img, units='MB') -> float:
    #     if units == 'MB':
    #         return (img.size * img.itemsize) / 1000000
    #     else:
    #         print('Unit conversion not supported.')

    def reshape(self, img) -> np.array:
        img_transposed = np.transpose(img, (1, 0, 2, 3))
        # [0][channel][z-stack][xdim][ydim]
        img_5d = img_transposed.reshape((1, self.n_channels, self.n_zstack, self.dim[0], self.dim[1]))
        return img_5d

    def load(self):
        _, extension = os.path.splitext(self.filename)
        if extension in VALID_EXTENSIONS:
            if extension == ".nd2":
                with nd2.ND2File(self.filename) as ndfile:
                    # self.size = (ndfile.size * ndfile.itemsize) / 1000000 # TODO: fix image size calculation
                    self.dim = (ndfile.sizes.get('X'), ndfile.sizes.get('Y'))
                    self.n_channels = ndfile.sizes.get('C')
                    self.n_zstack = ndfile.sizes.get('Z')
                    return self.reshape(ndfile.asarray())
            else:
                raise ValueError(".tif format not yet supported")
                # return cv2.imread(self.filename, 0)
        else:
            raise ValueError("Unsupported file format")