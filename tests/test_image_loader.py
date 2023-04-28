import unittest
import numpy as np
from PIL import Image
import tempfile
import os
from main import ImageLoader


class TestImageLoader(unittest.TestCase):
    def setUp(self):
        # Create a temporary TIFF image file for testing
        self.image = np.random.randint(0, 255, size=(512, 512)).astype(np.uint8)
        self.filename = tempfile.NamedTemporaryFile(suffix='.tif').name
        Image.fromarray(self.image).save(self.filename)

    def tearDown(self):
        # Delete the temporary image file
        os.remove(self.filename)

    def test_read_tif(self):
        # Test reading a TIFF image file
        reader = ImageLoader(self.filename)
        image_data = reader.image
        self.assertIsInstance(image_data, np.ndarray)
        self.assertEqual(image_data.shape, self.image.shape)
        self.assertTrue(np.array_equal(image_data, self.image))

    def test_read_unsupported_format(self):
        # Test reading an unsupported image file format
        with self.assertRaises(ValueError):
            reader = ImageLoader("image.jpg")