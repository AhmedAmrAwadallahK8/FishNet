import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
import src.user_interaction as usr_int
import tkinter as tk
from src.nodes.AbstractNode import AbstractNode
from nd2reader import ND2Reader
from segment_anything import sam_model_registry, SamPredictor
import src.image_processing as ip
from PIL import Image, ImageTk

class MSSGui():
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("600x600")
        self.root.title("Manual Sam Segmenter")

        img_arr = np.zeros((512,512,3)).astype(np.uint8)
        self.curr_img =  ImageTk.PhotoImage(image=Image.fromarray(img_arr))
        self.canvas = tk.Canvas(self.root, width=512, height=512)
        self.canvas.pack()
        self.canvas.create_image(20, 20, anchor="nw", image=self.curr_img)

        self.button_frame = tk.Frame(self.root)
        self.button_frame.columnconfigure(0, weight=1)
        self.button_frame.columnconfigure(1, weight=1)
        self.button_frame.columnconfigure(2, weight=1)

        self.done_button = tk.Button(self.button_frame, text="Done")
        self.done_button.grid(row=0, column=0, sticky=tk.W+tk.E)

        self.reset_button = tk.Button(self.button_frame, text="Reset")
        self.reset_button.grid(row=0, column=1, sticky=tk.W+tk.E)

        self.quit_button = tk.Button(self.button_frame, text="Quit")
        self.quit_button.grid(row=0, column=2, sticky=tk.W+tk.E)

        self.button_frame.pack(fill='x')


    def run(self):
        self.root.mainloop()

    def update_img(self, img_arr):
        img_arr = img_arr.astype(np.uint8)
        self.curr_img =  ImageTk.PhotoImage(image=Image.fromarray(img_arr))
        self.canvas.create_image(20, 20, anchor="nw", image=self.curr_img)


class ManualSamSegmenter(AbstractNode):
    def __init__(self):
        super().__init__(output_name="NucleusMask",
                         requirements=[],
                         user_can_retry=True,
                         node_title="Manual SAM Segmenter")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam_mask_generator = None
        self.sam = None
        self.sam_predictor = None
        self.input_boxes = [[]]
        self.gui = None

    def setup_sam(self):
        sam_checkpoint = "sam_model/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)
        self.sam_predictor = SamPredictor(self.sam)

    def process(self):
        self.gui.run()
        pass

    def initialize_node(self):
        raw_img = ip.get_raw_nucleus_img()
        self.prepared_img = ip.preprocess_img(raw_img)
        self.gui = MSSGui()
        self.gui.update_img(self.prepared_img)
        # self.setup_sam()

    def plot_output(self):
        pass
