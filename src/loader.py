import os
import cv2
import nd2
import uuid
import numpy as np 
# from .common import Node

# VALID_EXTENSIONS = [".tif", ".nd2"]


# class ImageContainer(Node):
#     def __init__(self, filename):
#         self.filename = filename
#         self.image = self.load()
#         self.size = None
#         self.dim = None
#         self.n_channels = None
#         self.n_zstack = None 

#     # def get_size(self, img, units='MB') -> float:
#     #     if units == 'MB':
#     #         return (img.size * img.itemsize) / 1000000
#     #     else:
#     #         print('Unit conversion not supported.')

#     def reshape(self, img) -> np.array:
#         img_transposed = np.transpose(img, (1, 0, 2, 3))
#         # [0][channel][z-stack][xdim][ydim]
#         img_5d = img_transposed.reshape((1, self.n_channels, self.n_zstack, self.dim[0], self.dim[1]))
#         return img_5d

#     def load(self):
#         _, extension = os.path.splitext(self.filename)
#         if extension in VALID_EXTENSIONS:
#             if extension == ".nd2":
#                 with nd2.ND2File(self.filename) as ndfile:
#                     # self.size = (ndfile.size * ndfile.itemsize) / 1000000 # TODO: fix image size calculation
#                     self.dim = (ndfile.sizes.get('X'), ndfile.sizes.get('Y'))
#                     self.n_channels = ndfile.sizes.get('C')
#                     self.n_zstack = ndfile.sizes.get('Z')
#                     return self.reshape(ndfile.asarray())
#             else:
#                 raise ValueError(".tif format not yet supported")
#                 # return cv2.imread(self.filename, 0)
#         else:
#             raise ValueError("Unsupported file format")

class Node:
    """
    The Node class represents a base node in a pipeline that processes input data and produces output data.

    Attributes:
    ----------
    id (UUID): A unique identifier for the node.
    name (str): An optional name for the node.
    execution_time (float): The execution time of the node in seconds.
    """

    def __init__(self, name=None):
        self.id = uuid.uuid4()
        self.name = name
        self.execution_time = None
        
    def check_valid_inputs(self, inputs):
        # Check if the inputs of the node match the expected inputs
        pass
    
    def check_valid_outputs(self, outputs):
        # Check if the outputs of the node match the expected outputs
        pass
    
    def process(self, inputs):
        # Define the processing API for the node
        pass


class ImageContainer:
    def __init__(self, image, metadata):
        self.image = image
        self.metadata = metadata

    def get_filename(self):
        return self.metadata['filename']

    def get_size(self, unit='MB'):
        size_in_bytes = self.image.nbytes
        if unit == 'B':
            return size_in_bytes
        elif unit == 'KB':
            return size_in_bytes / 1024
        elif unit == 'MB':
            return size_in_bytes / 1024 / 1024
        elif unit == 'GB':
            return size_in_bytes / 1024 / 1024 / 1024

class ND2Loader(Node):
    def __init__(self, filename, name="ND2Loader"):
        super().__init__(name=name)
        self.valid_inputs = None
        self.valid_outputs = ImageContainer
        self.valid_extensions = [".nd2"]
        self.inputs = None 
        self.outputs = None
        self.filename = filename

    def process(self, inputs):
        # Load the file and return an ImageContainer object
        if self.check_valid_inputs(inputs):
            image_data, metadata = self.load(self.filename)
            return ImageContainer(image_data, metadata)
        else:
            raise ValueError("Invalid inputs for ND2Loader")

    def load(self, filename):
        print('load')
        _, extension = os.path.splitext(filename)
        print(filename)
        if extension in self.valid_extensions:
            with nd2.ND2File(filename) as ndfile:
                image_data = self.reshape(ndfile.asarray())
                metadata = {
                    'filename': filename,
                    'size': image_data.nbytes,
                    'shape': image_data.shape,
                    'n_channels': ndfile.sizes.get('C'),
                    'n_zstack': ndfile.sizes.get('Z'),
                }
                return image_data, metadata
        else:
            raise ValueError("Unsupported file format")

    def reshape(self, img):
        img_transposed = np.transpose(img, (1, 0, 2, 3))
        # [0][channel][z-stack][xdim][ydim]
        img_5d = img_transposed.reshape((1, img.shape[1], img.shape[0], img.shape[2], img.shape[3]))
        return img_5d

    def check_valid_inputs(self, inputs):
        print('check valid inputs')
        # ND2Loader node should have no inputs
        return True if inputs == self.valid_inputs or len(inputs) == 0 else False

    
    def check_valid_outputs(self, outputs):
        print('check valid outputs')
        return isinstance(outputs, self.valid_outputs)