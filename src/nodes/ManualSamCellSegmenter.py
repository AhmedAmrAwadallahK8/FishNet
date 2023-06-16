# Two part process
# TODO
# Support Batch Processing
    # DONE Shift to batch processing
# Support Easily remove and add Rect
    # DONE
# Two Object Segmentation
    # DONE Need to decide if make a new GUI or continue to use the same
    # DONE Moving forward with single GUI
# Try to do everything in 1 GUI
    # So far so good...
# Update Reset
    # DONE
# Nucleus > Cytoplasm Logic Flow
    # DONE Base flow
    # DONE Need to post process results to link nuclei with cytoplasm
    # DONE Need to adjust cytoplasm to not include nuclei in segmentation
# Add ID on image
# Move Continue button to far right
# Add previous segmentation overlay button
# Move on to Dot Counting but still plenty to do after
# Support normal image size
# Togglable rect view
# Togglable previous selection view
    # For example let viewer see previous nucleus segmentation choices overlay
# Let user choose channel view

# OVERLAP PROBLEM
# NUCLEUS EXISTING IN TWO CYTOS PROBLEM

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
import src.sam_processing as sp
from PIL import Image, ImageTk

"""
ManualSamCellSegmenter output format is a length 2 dictionary.
Use the key ManualCellMaskPack feth this data
The two keys within ManualCellMaskPack are below
Contains Key nucleus which contains nucleus mask id data
Contains Key cytoplasm which contains cytoplasm mask id data

"""

class RectTracker:
    def __init__(self, canvas, gui, box_tag):
        self.canvas = canvas
        self.rect_count = 1
        self.gui = gui
        self.box_tag = box_tag
        self.item = None
        self.box = None
        self.boxes = []
		
    def draw(self, start, end, **opts):
        return self.canvas.create_rectangle(*(list(start)+list(end)), **opts)
		
    def autodraw(self, **opts):
        """Setup automatic drawing; supports command option"""
        self.start = None
        self.canvas.bind("<Button-1>", self.__update, '+')
        self.canvas.bind("<B1-Motion>", self.__update, '+')
        self.canvas.bind("<ButtonRelease-1>", self.__stop, '+')
        self._command = opts.pop('command', lambda *args: None)
        self.rectopts = opts

    def __update(self, event):
        if not self.start:
            self.start = [event.x, event.y]
            return
        if self.item is not None:
            self.canvas.delete(self.item)
        self.item = self.draw(
            self.start,
            (event.x, event.y),
            tags=(self.box_tag),
            **self.rectopts)
        self._command(self.start, (event.x, event.y))
	
    def __stop(self, event):
        self.start = None
        self.item = None
        # self.canvas.delete(self.item)
        # self.give_final_box()

    def give_final_box(self):
        pass
        # self.gui.segment_box(self.box)
	
    def get_box(self, start, end, tags=None, ignoretags=None, ignore=[]):
        xlow = min(start[0], end[0])
        xhigh = max(start[0], end[0])
	
        ylow = min(start[1], end[1])
        yhigh = max(start[1], end[1])
	
        self.box = [xlow, ylow, xhigh, yhigh]

    def mouse_hit_test(self, pos, tags=None, ignoretags=None, ignore=[]):
        def get_area(rect):
            xlow, ylow, xhigh, yhigh = self.canvas.coords(rect)
            return (xhigh-xlow)*(yhigh-ylow)
        ignore = set(ignore)
        ignore.update([self.item])
		
        if isinstance(tags, str):
            tags = [tags]
		
        if tags:
            tocheck = []
            for tag in tags:
                tocheck.extend(self.canvas.find_withtag(tag))
        else:
            tocheck = self.canvas.find_all()
        tocheck = [x for x in tocheck if x != self.item]
        if ignoretags:
            if not hasattr(ignoretags, '__iter__'):
                ignoretags = [ignoretags]
            tocheck = [x for x in tocheck if x not in self.canvas.find_withtag(it) for it in ignoretags]
		
        self.items = tocheck
        items = []
        for item in tocheck:
            if item not in ignore:
                xlow, ylow, xhigh, yhigh = self.canvas.coords(item)
                x, y = pos[0], pos[1]
                if (xlow < x < xhigh) and (ylow < y < yhigh):
                    items.append(item)
        smallest_item = None
        smallest_area = 0
        first = True
        for item in items:
            if len(items) < 1:
                break
            if len(items) == 1:
                smallest_item = item
                break
            if first:
                first = False
                smallest_item = item
                smallest_area = get_area(item)
            else:
                curr_area = get_area(item)
                if curr_area < smallest_area:
                    smallest_item = item
                    smallest_area = curr_area
        return smallest_item

class MSSGui():
    def __init__(self, owner, canv_height, canv_width):
        from src.fishnet import FishNet
        self.channel_states = {}
        self.z_states = {}
        for k in FishNet.channel_meta.keys():
            self.channel_states[k] = False
        for k in FishNet.z_meta.keys():
            self.z_states[k] = False
        self.canv_width = canv_width
        self.canv_height = canv_height
        self.app_width = int(canv_width*1.2)
        self.app_height = int(canv_width*1.2)
        self.image_reps = 3
        self.curr_rep = 0
        self.master_node = owner
        self.root = tk.Tk()
        self.root.geometry(f"{self.app_width}x{self.app_height}")
        self.root.title("Nucleus Selection Step")
        self.box_tag = "box"

        img_arr = np.zeros((self.canv_height, self.canv_width,3)).astype(np.uint8)
        self.curr_img =  ImageTk.PhotoImage(image=Image.fromarray(img_arr))
        self.canvas = tk.Canvas(self.root, width=self.canv_width, height=self.canv_height)
        self.canvas.pack()
        self.rect = RectTracker(self.canvas, self, self.box_tag)
        def on_drag(start, end):
            self.rect.get_box(start, end)
        self.rect.autodraw(fill="", width=2, command=on_drag)
        
        self.img_container = self.canvas.create_image(0, 0, anchor="nw", image=self.curr_img)

        self.button_frame = tk.Frame(self.root)
        self.button_frame.columnconfigure(0, weight=1)
        self.button_frame.columnconfigure(1, weight=1)
        self.button_frame.columnconfigure(2, weight=1)
        self.button_frame.columnconfigure(3, weight=1)
        self.button_frame.columnconfigure(4, weight=1)
        self.button_frame.columnconfigure(5, weight=1)

        self.continue_button = tk.Button(self.button_frame,
                                         text="Continue",
                                         command=self.continue_program)
        self.continue_button.grid(row=0, column=0, sticky=tk.W+tk.E)

        self.reset_button = tk.Button(self.button_frame,
                                      text="Reset",
                                      command=self.reset)
        self.reset_button.grid(row=0, column=1, sticky=tk.W+tk.E)

        self.segment_view_button = tk.Button(self.button_frame,
                                     text="Segment View",
                                     command=self.segment_view)
        self.segment_view_button.grid(row=0, column=2, sticky=tk.W+tk.E)

        self.default_view_button = tk.Button(self.button_frame,
                                     text="Default View",
                                     command=self.default_view)
        self.default_view_button.grid(row=0, column=3, sticky=tk.W+tk.E)

        self.segment_button = tk.Button(self.button_frame,
                                     text="Segment Image",
                                     command=self.segment)
        self.segment_button.grid(row=0, column=4, sticky=tk.W+tk.E)

        self.nuc_overlay_btn = tk.Button(self.button_frame,
                                     text="Nucleus Overlay",
                                     command=self.nuc_overlay,
                                     bg="red")
        self.nuc_overlay_btn.grid(row=0, column=5, sticky=tk.W+tk.E)


        # Channel Buttons
        self.channel_buttons = {}
        channel_row = 1
        col = 0
        for chan_k in self.channel_states.keys():
            btn = tk.Button(
                self.button_frame,
                text=f"Experi: {chan_k}",
                command=lambda m=chan_k: self.chan_adjustment(m),
                bg="red")
            btn.grid(row=channel_row, column=col, sticky=tk.W+tk.E)
            self.channel_buttons[chan_k] = btn
            col += 1

        # Z Buttons
        self.z_buttons = {}
        z_row = 2
        col = 0
        for z_k in self.z_states.keys():
            btn = tk.Button(
                self.button_frame,
                text=f"Z: {z_k}",
                command=lambda m=z_k: self.z_adjustment(m),
                bg="red")
            btn.grid(row=z_row, column=col, sticky=tk.W+tk.E)
            self.z_buttons[z_k] = btn
            col += 1
            
            

        self.button_frame.pack(fill='x')
        self.curr_view = "default"
        self.canvas.bind('<Motion>', self.on_mouse_over, '+')
        self.canvas.bind('<Button-3>', self.on_click, '+')
        self.overlay_with_nuc_seg = False

    def z_adjustment(self, btn_name):
        btn = self.z_buttons[btn_name]
        self.z_states[btn_name] = not self.z_states[btn_name] 
        if self.z_states[btn_name]:
            btn.configure(bg = "green")
        else:
            btn.configure(bg = "red")
        self.master_node.reinitialize_base_img(self.channel_states, self.z_states)
        self.reset()

    def chan_adjustment(self, btn_name):
        btn = self.channel_buttons[btn_name]
        self.channel_states[btn_name] = not self.channel_states[btn_name] 
        if self.channel_states[btn_name]:
            btn.configure(bg = "green")
        else:
            btn.configure(bg = "red")
        self.master_node.reinitialize_base_img(self.channel_states, self.z_states)
        self.reset()

    def nuc_overlay(self):
        if self.curr_rep == 1: #Cyto segmentation step
            self.overlay_with_nuc_seg = not self.overlay_with_nuc_seg
            self.refresh_view()
        else:
            print("This toggle only works on the cytoplasm segmentation step")

        if self.overlay_with_nuc_seg:
            self.nuc_overlay_btn.configure(bg = "green")
        else:
            self.nuc_overlay_btn.configure(bg = "red")


    def get_bboxes(self):
        bboxes = []
        boxes = []
        boxes.extend(self.canvas.find_withtag(self.box_tag))
        for box in boxes:
            bboxes.append(self.canvas.coords(box))
        return bboxes

    def remove_tiny_bboxes(self, bboxes):
        min_area = 100
        bboxes_pruned = []
        for bbox in bboxes:
            bbox_area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
            if bbox_area > min_area:
                bboxes_pruned.append(bbox)
        return bboxes_pruned
          

    def segment(self):
        if self.curr_rep == 2:
            print("This button only works on a segmentation step")
            return
        bboxes = self.get_bboxes()
        bboxes = self.remove_tiny_bboxes(bboxes)
        self.master_node.update_bboxes(bboxes)
        self.master_node.process_img()
        self.refresh_view()
        # img_arr = self.master_node.get_curr_img()
        # self.update_img(img_arr)

    def refresh_view(self):
        if self.curr_view == "default":
            img_arr = self.master_node.get_curr_img()
            if self.overlay_with_nuc_seg:
                nuc_seg = self.master_node.get_nuc_seg()
                img_arr = np.where(nuc_seg > 0, nuc_seg, img_arr)
            self.update_img(img_arr)
        elif self.curr_view == "segment":
            img_arr = self.master_node.get_segment_img()
            self.update_img(img_arr)

    def reset(self):
        self.master_node.soft_reset()
        self.refresh_view()
        self.remove_all_boxes()

    def continue_program(self):
        if self.curr_rep < 2:
            bboxes = self.get_bboxes()
            self.master_node.update_bboxes(bboxes)
            mask_class = self.get_mask_class_from_user()
            self.master_node.produce_and_store_mask(mask_class)

        self.curr_rep += 1

        if self.curr_rep == 1:
            self.root.title("Cytoplasm Selection Step")
        elif self.curr_rep == 2:
            self.root.title("Select Base Image for Saving")
            self.overlay_with_nuc_seg = False

        if self.curr_rep == self.image_reps:
            self.master_node.set_valid_gui_exit()
            self.exit_gui()
        else:
            self.reset()

    def get_mask_class_from_user(self):
        if self.curr_rep == 0:
            return "nucleus"
        elif self.curr_rep == 1:
            return "cytoplasm"

    def exit_gui(self):
        self.root.destroy()

    def remove_all_boxes(self):
        boxes = []
        boxes.extend(self.canvas.find_withtag(self.box_tag))
        for box in boxes:
            self.canvas.delete(box)
        
    def segment_view(self):
        if self.curr_rep == 2:
            print("This button only works on a segmentation step")
            return
        self.curr_view = "segment"
        self.refresh_view()
        # img_arr = self.master_node.get_segment_img()
        # self.update_img(img_arr)
        
    def default_view(self):
        self.curr_view = "default"
        self.refresh_view()
        # img_arr = self.master_node.get_curr_img()
        # self.update_img(img_arr)

    def segment_box(self, box):
        pass
        # self.master_node.updates_boxes(box)
        # self.master_node.process_img()

    def run(self):
        self.root.mainloop()

    def update_img(self, img_arr):
        img_arr = img_arr.astype(np.uint8)
        self.curr_img =  ImageTk.PhotoImage(image=Image.fromarray(img_arr))
        self.canvas.itemconfig(self.img_container, image=self.curr_img)
        # self.canvas.create_image(20, 20, anchor="nw", image=self.curr_img)

    def on_click(self, event):
        x = event.x
        y = event.y
        selected_rect = self.rect.mouse_hit_test([x,y], tags=[self.box_tag])
        if selected_rect is not None:
            self.canvas.delete(selected_rect)

    def on_mouse_over(self, event):
        x = event.x
        y = event.y
        selected_rect = self.rect.mouse_hit_test([x,y], tags=[self.box_tag])
        for sub_rect in self.rect.items:
            if sub_rect is not selected_rect:
                self.canvas.itemconfig(sub_rect, outline='black')
            else:
                self.canvas.itemconfig(sub_rect, outline='red')


class ManualSamCellSegmenter(AbstractNode):
    def __init__(self):
        super().__init__(output_name="ManualCellMaskPack",
                         requirements=[],
                         user_can_retry=False,
                         node_title="Manual SAM Cell Segmenter")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.image_size = 1024
        self.targ_pixel_area = 768*768
        self.sam_mask_generator = None
        self.sam = None
        self.sam_predictor = None
        self.input_boxes = []
        self.input_points = [[0,0]]
        self.input_labels = [0]
        self.gui = None
        self.default_size_img = None
        self.prepared_img = None
        self.prev_prepared_img = None
        self.curr_img = None
        self.segment_img = None
        self.nuc_class = "nucleus"
        self.cyto_class = "cytoplasm"
        self.output_pack = {self.nuc_class: None, self.cyto_class: None}
        self.valid_gui_exit = False

    def set_valid_gui_exit(self):
        self.valid_gui_exit = True

    def get_nuc_seg(self):
        nuc_id_mask = self.output_pack[self.nuc_class]
        nuc_seg = ip.generate_single_colored_mask(nuc_id_mask, color=(0, 0, 255))
        return nuc_seg

    def pop_boxes(self):
        if len(self.input_boxes) > 0:
            self.input_boxes.pop()

    def produce_and_store_mask(self, mask_class):
        self.sam_predictor.set_image(self.prepared_img)
        sam_masks = self.apply_sam_pred()
        mask_img =  sp.generate_mask_img_manual(self.prepared_img, sam_masks)
        self.output_pack[mask_class] = mask_img
        

    def setup_sam(self):
        sam_checkpoint = "sam_model/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)
        self.sam_predictor = SamPredictor(self.sam)
        self.sam_predictor.set_image(self.prepared_img)

    def soft_reset(self):
        self.input_boxes = []
        self.curr_img = self.prepared_img.copy()
        self.segment_img = np.zeros(self.prepared_img.shape)

    def reset_boxes(self):
        self.input_boxes = []

    def gui_update_img(self):
        self.gui.update_img(self.curr_img)

    def update_bboxes(self, bboxes):
        self.input_boxes = bboxes

    def push_box(self, box):
        self.input_boxes.append(box)

    def get_curr_img(self):
        return self.curr_img

    def get_segment_img(self):
        return self.segment_img

    def process_img(self):
        if len(self.input_boxes) == 0:
            self.curr_img = self.prepared_img.copy()
            self.segment_img = np.zeros(self.prepared_img.shape)
            return
        if np.array_equal(self.prepared_img, self.prev_prepared_img):
            pass
        else:
            self.prev_prepared_img = self.prepared_img.copy()
            self.sam_predictor.set_image(self.prepared_img)
        
        sam_masks = self.apply_sam_pred()
            
        mask_img =  sp.generate_mask_img_manual(self.prepared_img, sam_masks)
        self.segment_img = ip.generate_colored_mask(mask_img)
        contour_img = ip.generate_advanced_contour_img(mask_img)
        anti_ctr = ip.generate_anti_contour(contour_img).astype(np.uint8)
        self.curr_img = self.prepared_img.astype(np.uint8)
        self.curr_img *= anti_ctr
        self.curr_img += contour_img

    def apply_sam_pred(self):
        tensor_boxes = torch.tensor(self.input_boxes, device=self.device)
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            tensor_boxes, self.prepared_img.shape[:2])
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False)
        masks = masks.cpu().numpy()
        return masks

    def stitch_cells(self):
        temp_nuc_id = -1
        stitched_nuc_id_mask = None
        nuc_id_mask = self.output_pack[self.nuc_class]
        stitched_nuc_id_mask = nuc_id_mask.copy()
        cyto_id_mask = self.output_pack[self.cyto_class]
        nuc_activation = np.where(nuc_id_mask > 0, 1, 0)

        cyto_nuc_activated = cyto_id_mask * nuc_activation
        valid_cytos = np.unique(cyto_nuc_activated)
        for master_cyto_id in valid_cytos:
            if master_cyto_id == 0:
                continue
            cyto_id_activation = np.where(cyto_id_mask == master_cyto_id, 1, 0)
            nuc_cyto_id_activated = cyto_id_activation*nuc_id_mask
            child_nuc_id = np.unique(nuc_cyto_id_activated)[1]
            # Code below is likely useless
            # master_cyto_id should just by cyto_id
            # nuc_id_activation = np.where(nuc_id_mask == child_nuc_id, 1, 0)
            # cyto_nuc_id_activated = nuc_id_activation*cyto_id_mask
            # master_cyto_id = np.unique(cyto_nuc_id_activated)[1]
            # Code above is likely useless
            id_collision_sum = np.sum(
                np.where(
                    nuc_id_mask == master_cyto_id,
                    1,
                    0
                )
            )
            # Check if a nucleus already has a cytoplasm id
            # if it does then handle it
            if id_collision_sum == 0:
                stitched_nuc_id_mask = np.where(
                    stitched_nuc_id_mask == child_nuc_id,
                    master_cyto_id,
                    stitched_nuc_id_mask)
            else:
                stitched_nuc_id_mask = np.where(
                     stitched_nuc_id_mask == master_cyto_id,
                     temp_nuc_id,
                     stitched_nuc_id_mask)
                stitched_nuc_id_mask = np.where(
                     stitched_nuc_id_mask == child_nuc_id,
                     master_cyto_id,
                     stitched_nuc_id_mask)
                stitched_nuc_id_mask = np.where(
                     stitched_nuc_id_mask == temp_nuc_id,
                     child_nuc_id,
                     stitched_nuc_id_mask)
        self.output_pack[self.nuc_class] = stitched_nuc_id_mask
        return True

    def remove_nucleus_from_cytoplasm_mask(self, after_stitch):
        if not after_stitch:
            return
        nuc_id_mask = self.output_pack[self.nuc_class]
        cyto_id_mask = self.output_pack[self.cyto_class]
        anti_nuc_activation = np.where(nuc_id_mask > 0, 0, 1)
        updated_cyto_id_mask = cyto_id_mask * anti_nuc_activation
        self.output_pack[self.cyto_class] = updated_cyto_id_mask
        # Some debugging code for stitching
        # output_compare = np.hstack((updated_cyto_id_mask, nuc_id_mask))
        # plt.figure(figsize=(12,8))
        # plt.axis('off')
        # plt.imshow(output_compare)
        # plt.show()
        
        

    def process(self):
        self.gui.run()
        if self.valid_gui_exit:
            self.set_node_as_successful()
            stitch_compelete = self.stitch_cells()
            self.remove_nucleus_from_cytoplasm_mask(stitch_compelete)
        del self.sam
        del self.sam_mask_generator
        del self.sam_predictor
        torch.cuda.empty_cache()
        

    def hello_world(self):
        print("Hello World")

    def translate_state_into_index(self, state_dict, ind_dict):
        ind_list = []
        for state_k in state_dict.keys():
            state = state_dict[state_k]
            if state:
                ind_list.append(ind_dict[state_k])
        return ind_list

    # Currently Assuming canvas doesnt need to change
    def reinitialize_base_img(self, channel_states, z_states):
        from src.fishnet import FishNet
        channels = self.translate_state_into_index(channel_states, FishNet.channel_meta)
        z_axi = self.translate_state_into_index(z_states, FishNet.z_meta)
        raw_img = ip.get_specified_channel_combo_img(channels, z_axi)
        if raw_img.sum() == 0:
            raw_img = raw_img[:, :, np.newaxis]
            zero_img = raw_img.copy()
            raw_img = np.append(raw_img, zero_img, axis=2)
            raw_img = np.append(raw_img, zero_img, axis=2)
            self.prepared_img = raw_img.copy()
            self.default_size_img = raw_img.copy()
        else:
            self.prepared_img = ip.preprocess_img2(raw_img)
            self.default_size_img = self.prepared_img.copy()

        self.prepared_img = ip.resize_img_to_pixel_size(
            self.prepared_img,
            self.targ_pixel_area)
        self.curr_img = self.prepared_img.copy()
        self.segment_img = np.zeros(self.prepared_img.shape)
        

    def initialize_node(self):
        # raw_img = ip.get_all_channel_img()
        # self.prepared_img = ip.preprocess_img2(raw_img)
        # self.default_size_img = self.prepared_img.copy()

        
        zero_img = ip.get_zerod_img()
        zero_img = zero_img[:, :, np.newaxis]
        raw_img = zero_img.copy()
        raw_img = np.append(raw_img, zero_img, axis=2)
        raw_img = np.append(raw_img, zero_img, axis=2)
        self.prepared_img = raw_img.copy()
        self.default_size_img = self.prepared_img.copy()
        self.prepared_img = ip.resize_img_to_pixel_size(
            self.prepared_img,
            self.targ_pixel_area)
        self.curr_img = self.prepared_img.copy()
        self.segment_img = np.zeros(self.prepared_img.shape)
        canv_height, canv_width, _=self.prepared_img.shape
        self.gui = MSSGui(self, canv_height, canv_width)
        self.gui.update_img(self.prepared_img)
        self.setup_sam()

    def plot_output(self):
        pass

    def save_output(self):
        base_shape = self.default_size_img.shape
        base_height = base_shape[0]
        base_width = base_shape[1]
        targ_pixel_area = base_shape[0]*base_shape[1]
        base_img = self.default_size_img.copy()
        outline_img = None
        segment_img = None
        segment_overlay = None

        nuc_id_mask = self.output_pack[self.nuc_class]
        cyto_id_mask = self.output_pack[self.cyto_class]

        nuc_segment_img = ip.generate_colored_mask(nuc_id_mask)
        cyto_segment_img = ip.generate_colored_mask(cyto_id_mask)
        nuc_segment_img = ip.resize_img(nuc_segment_img, base_height, base_width, "linear")
        cyto_segment_img = ip.resize_img(cyto_segment_img, base_height, base_width, "linear")
        segment_img = nuc_segment_img + cyto_segment_img
        segment_overlay = base_img.copy()*0.5 + segment_img*0.5

        nuc_contour = ip.generate_advanced_contour_img(nuc_id_mask)
        cyto_contour = ip.generate_advanced_contour_img(cyto_id_mask)
        nuc_contour = ip.resize_img(nuc_contour, base_height, base_width, "linear")
        cyto_contour = ip.resize_img(cyto_contour, base_height, base_width, "linear")
        outline_img = np.where(nuc_contour > 0, 255, base_img)
        outline_img = np.where(cyto_contour > 0, 255, outline_img)
        outline_img = ip.add_label_to_img(outline_img, cyto_id_mask)

        nuc_activation = np.where(nuc_segment_img > 0, 1, 0)
        cyto_activation = np.where(cyto_segment_img > 0, 1, 0)
        cell_activation = nuc_activation + cyto_activation

        # For presentation largely
        segment_overlay_activated = segment_overlay*cell_activation
        outline_activated = outline_img*cell_activation
        # base_img_cell_activated = self.prepared_img*cell_activation
        # base_img_nuc_activated = self.prepared_img*nuc_activation
        # base_img_cyto_activated = self.prepared_img*cyto_activation
        # self.save_img(base_img_cell_activated, "base_img_cell_activated.png")
        # self.save_img(base_img_nuc_activated, "base_img_nuc_activated.png")
        # self.save_img(base_img_cyto_activated, "base_img_cyto_activated.png")

        # SPECIFIC PRESENTATION CODE, NOT FOR PRODUCTION
        # cyto_id_mask_box = cyto_id_mask[:, :, np.newaxis]
        # offset = 20
        # cell_box = nuc_activation+cyto_activation
        # cell_box = cell_box.astype(np.uint8)
        # contours, hierarchy = cv2.findContours(cell_box,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        # x,y,w,h = cv2.boundingRect(contours[0])
        # bbox = [x, y, x+w, y+h]
        # xmin = x - offset
        # ymin = y - offset
        # xmax = x+w + offset
        # ymax = y+h + offset
        # print(bbox)
        # base_img_crop = self.prepared_img[ymin:ymax, xmin:xmax, :]
        # nuc_only_segmented = self.prepared_img*np.where(nuc_activation == 1, 0, 1)
        # nuc_only_segmented = nuc_only_segmented + nuc_segment_img
        # nuc_only_segmented = nuc_only_segmented[ymin:ymax, xmin:xmax, :]
        # both_segmented = self.prepared_img*np.where(nuc_activation == 1, 0, 1)
        # both_segmented = both_segmented*np.where(cyto_activation == 1, 0, 1)
        # both_segmented = both_segmented + nuc_segment_img + cyto_segment_img
        # both_segmented = both_segmented[ymin:ymax, xmin:xmax, :]
        # self.save_img(base_img_crop, "base_cell_crop.png")
        # self.save_img(nuc_only_segmented, "base_nuc_crop.png")
        # self.save_img(both_segmented, "base_nuc_cyto_crop.png")
        
        

        self.save_img(segment_img, "manual_cell_segment.png")
        self.save_img(segment_overlay, "manual_cell_overlay.png")
        self.save_img(outline_img, "manual_cell_outline.png")
        self.save_img(segment_overlay_activated, "manual_cell_overlay_activated.png")
        self.save_img(outline_activated, "manual_cell_outline_activated.png")
        self.save_img(nuc_segment_img, "manual_nuc_segment.png")
        self.save_img(cyto_segment_img, "manual_cyto_segment.png")
