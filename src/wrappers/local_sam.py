import torch
import torchvision
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

class LocalSam():
    """
    A wrapper class for expected SAM functionality

    Global Variables:
        Nothing

    Global Function:
        Nothing

    Attributes:
        device (str): device to utilize for model computation. Use gpu if
        available otherwise cpu
        sam_checkpoint (str): specific model checkpoint
        model_type (str): type of SAM model
        base_model_loaded (boolean): boolean to see if model is loaded
        augmented_model_initialized (boolean): boolean to see if the 
        augmented model is initialized
        auto_model_initialized (boolean): boolean to see if the pure SAM
        model is loaded
        sam_predictor (SamPredictor): SAM object associated with working
        with user input to produce a mask
        sam (pytorch_tensor?): the SAM model
        mask_generator (SamAutomaticMaskGenerator): SAM object associated with
        pure SAM mask output
        context_img (ndarray): Augmented SAM requires a context image to apply
        its prediction

    Methods:
        load_base_model(): loads the model checkpoint
        setup_auto_mask_pred(default_sam_settings): initalizes pure SAM
        setup_augmented_mask_pred(): initializes augmented SAM
        get_augmented_mask_pred(raw_boxes): given raw_boxes outputs a mask 
        within the mask and context_img
        get_auto_mask_pred(img): given an img outputs any mask the model feels
        confident in
        set_image_context(img): sets the image that an augmented prediction
        operates on
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam_checkpoint = "sam_model/sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"
        self.base_model_loaded = False
        self.augmented_model_initialized = False
        self.auto_model_initialized = False
        self.sam_predictor = None
        self.sam = None
        self.mask_generator = None
        self.context_img = None

    def load_base_model(self):
        """
        Loads the sam checkpoint if it hasn't already been loaded

        Args:
            Nothing

        Returns:
            Nothing
        """
        if self.base_model_loaded:
            return
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device=self.device)
        self.base_model_loaded = True

    def setup_auto_mask_pred(self, default_sam_settings):
        """
        Initializes pure SAM if it hasnt been initialized.
        If the base model is not loaded it will load base model before
        initalization

        Args:
            Nothing

        Returns:
            Nothing
        """
        if not self.base_model_loaded:
            self.load_base_model()
        if self.auto_model_initialized:
            return
        self.mask_generator = SamAutomaticMaskGenerator(model=self.sam, **default_sam_settings)
        self.auto_model_initialized = True

    def setup_augmented_mask_pred(self):
        """
        Initializes augmented SAM if it hasnt been initialized.
        If the base model is not loaded it will load base model before
        initalization

        Args:

        Returns:
        """
        if not self.base_model_loaded:
            self.load_base_model()
        if self.augmented_model_initialized:
            return
        self.sam_predictor = SamPredictor(self.sam)
        self.augmented_model_initialized = True

    def get_augmented_mask_pred(self, raw_boxes):
        """
        Converts list of bboxes into a tensor containing properly transformed
        torch boxes. The torch boxes and the current context_img are then
        used as inputs to SAM for prediction.

        Args:
            raw_boxes (list): list of bbox coordinates

        Returns:
            list: list of segmentation instances
        """
        tensor_boxes = torch.tensor(raw_boxes, device=self.device)
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            tensor_boxes, self.context_img.shape[:2])
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False)
        masks = masks.cpu().numpy()
        return masks

    def get_auto_mask_pred(self, img):
        """
        Directly feeds an image into SAM for segmentation prediction and 
        returns segmentation information

        Args:
            img (ndarray): image data within numpy array

        Returns:
            list: list of dictionaries that provides a variety of information
            pertaining to a single segmentation instance
        """
        mask = self.mask_generator.generate(img)
        return mask

    def set_image_context(self, img):
        """
        Sets the current context_img

        Args:
            img (ndarray): image data within numpy array

        Returns:
            Nothing
        """
        self.context_img = img
        self.sam_predictor.set_image(img)
