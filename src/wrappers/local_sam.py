import torch
import torchvision
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

class LocalSam():
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

    def load_base_model(self):
        if self.base_model_loaded:
            return
        self.sam = sam_model_registry[self.model_type](self.checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)
        self.base_model_loaded = True

    def setup_auto_mask_gen(self, default_sam_settings):
        if not self.base_model_loaded:
            self.load_base_model()
        if self.auto_model_initialized:
            return
        self.mask_generator = SamAutomaticMaskGenerator(model=self.sam, **default_sam_settings)
        self.auto_model_initialized = True

    def setup_augmented_mask(self):
        if not self.base_model_loaded:
            self.load_base_model()
        if self.augmented_model_initialized:
            return
        self.sam_predictor = SamPredictor(self.sam)
        self.augmented_model_initialized = True

    def get_augmented_mask_pred(self, boxes):
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=boxes,
            multimask_output=False)
        masks = masks.cpu().numpy()
        return masks

    def get_auto_mask_pred(self, img):
        mask = self.mask_generator.generate(img)
        return mask

    def set_image_context(self, img):
        self.sam_predictor.set_image(img)
