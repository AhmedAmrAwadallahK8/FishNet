import numpy as np
import os
import torch
import torchvision
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
Use the key ManualCellMaskPack fetch the data
The two keys within ManualCellMaskPack are below
Contains Key nucleus which contains nucleus mask id data
Contains Key cytoplasm which contains cytoplasm mask id data

"""

class ManualSamCellSegmenter(AbstractNode):
    """

    Global Variables:
    Global Functions:
    Attributes:
    Methods:
    """
    def __init__(self):
        from src.fishnet import FishNet
        super().__init__(output_name="ManualCellMaskPack",
                         requirements=[],
                         user_can_retry=False,
                         node_title="Manual SAM Cell Segmenter")
        self.targ_pixel_area = 768*768
        self.cell_figures_folder = "cell_figures/"
        self.id_masks_folder = "id_masks/"
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
        self.cell_figures_path = FishNet.save_folder + self.cell_figures_folder
        self.id_mask_path = FishNet.save_folder + self.id_masks_folder
        self.create_folder(self.cell_figures_folder)
        self.create_folder(self.id_masks_folder)

    def process(self):
        self.gui.run()
        if self.valid_gui_exit:
            self.set_node_as_successful()
            stitch_compelete = self.stitch_cells()
            self.remove_nucleus_from_cytoplasm_mask(stitch_compelete)
            self.remove_nuclei_with_no_cyto()
            self.remove_cyto_with_no_nuclei() #Has to be after nucleus
            self.reset_id_sequence()
            cyto_id_mask = self.output_pack[self.cyto_class]
            cell_count = np.max(cyto_id_mask)
            print(f"Total Valid Cells Found: {cell_count:d}")

    def initialize_node(self):
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


    def create_folder(self, folder_name):
        from src.fishnet import FishNet
        save_folder = FishNet.save_folder + folder_name
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

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
        from src.fishnet import FishNet
        if np.array_equal(self.prepared_img, self.prev_prepared_img):
            pass
        else:
            self.prev_prepared_img = self.prepared_img.copy()
            FishNet.sam_model.set_image_context(self.prepared_img)
        sam_masks = self.apply_sam_pred()
        mask_img =  sp.generate_mask_img_manual(self.prepared_img, sam_masks)
        self.output_pack[mask_class] = mask_img
        

    def setup_sam(self):
        from src.fishnet import FishNet
        FishNet.sam_model.setup_augmented_mask_pred()
        FishNet.sam_model.set_image_context(self.prepared_img)

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
        from src.fishnet import FishNet
        if len(self.input_boxes) == 0:
            self.curr_img = self.prepared_img.copy()
            self.segment_img = np.zeros(self.prepared_img.shape)
            return
        if np.array_equal(self.prepared_img, self.prev_prepared_img):
            pass
        else:
            self.prev_prepared_img = self.prepared_img.copy()
            FishNet.sam_model.set_image_context(self.prepared_img)
        
        sam_masks = self.apply_sam_pred()
            
        mask_img =  sp.generate_mask_img_manual(self.prepared_img, sam_masks)
        self.segment_img = ip.generate_colored_mask(mask_img)
        contour_img = ip.generate_advanced_contour_img(mask_img)
        anti_ctr = ip.generate_anti_contour(contour_img).astype(np.uint8)
        self.curr_img = self.prepared_img.astype(np.uint8)
        self.curr_img *= anti_ctr
        self.curr_img += contour_img

    def apply_sam_pred(self):
        from src.fishnet import FishNet
        masks = FishNet.sam_model.get_augmented_mask_pred(self.input_boxes)
        return masks

    def get_likely_child_nuc_id(self, nuc_cyto_id_activated):
        possible_child_nuc_ids = np.unique(nuc_cyto_id_activated)
        first = True
        best_nuc_id = -1
        largest_nuc_area = -1
        for nuc_id in possible_child_nuc_ids:
            if nuc_id == 0:
                continue
            nuc_isolated = np.where(nuc_cyto_id_activated == nuc_id, 1, 0)
            nuc_area = np.sum(nuc_isolated)
            if first:
                first = False
                largest_nuc_area = nuc_area
                best_nuc_id = nuc_id
            else:
                if nuc_area > largest_nuc_area:
                    largest_nuc_area = nuc_area
                    best_nuc_id = nuc_id
        return best_nuc_id, largest_nuc_area

    def stitch_cells(self):
        temp_nuc_id = -1
        stitched_nuc_id_mask = None
        nuc_id_mask = self.output_pack[self.nuc_class]
        stitched_nuc_id_mask = nuc_id_mask.copy()
        cyto_id_mask = self.output_pack[self.cyto_class]
        nuc_activation = np.where(nuc_id_mask > 0, 1, 0)

        cyto_nuc_activated = cyto_id_mask * nuc_activation
        valid_cytos = np.unique(cyto_nuc_activated)

        cyto_nuc_clues = {}

        # need to search for best nuc ids for each cyto and then assign nuc to cyto
        for master_cyto_id in valid_cytos:
            if master_cyto_id == 0:
                continue
            cyto_id_activation = np.where(cyto_id_mask == master_cyto_id, 1, 0)
            nuc_cyto_id_activated = cyto_id_activation*nuc_id_mask
            cyto_nuc_clues[master_cyto_id] = self.get_likely_child_nuc_id(
                nuc_cyto_id_activated)

        confirmed_nuc_area = {}
        confirmed_nuc_id = {}

        for master_cyto_id in cyto_nuc_clues.keys():
            if master_cyto_id == 0:
                continue
            child_nuc_id, child_nuc_area = cyto_nuc_clues[master_cyto_id]
            if master_cyto_id in confirmed_nuc_area.keys():
                if child_nuc_area > confirmed_nuc_area[master_cyto_id]:
                    confirmed_nuc_area[child_nuc_id] = child_nuc_area
                    confirmed_nuc_id[child_nuc_id] = master_cyto_id
                else:
                    continue
            else:
                confirmed_nuc_area[child_nuc_id] = child_nuc_area
                confirmed_nuc_id[child_nuc_id] = master_cyto_id

        for child_nuc_id in confirmed_nuc_id.keys():
            master_cyto_id = confirmed_nuc_id[child_nuc_id]
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

    def remove_nuclei_with_no_cyto(self):
        nuc_id_mask = self.output_pack[self.nuc_class]
        cyto_id_mask = self.output_pack[self.cyto_class]
        max_cyto_id = np.max(cyto_id_mask)
        updated_nuc_id_mask = np.where(nuc_id_mask > max_cyto_id, 0, nuc_id_mask)
        self.output_pack[self.nuc_class] = updated_nuc_id_mask
        
    def remove_cyto_with_no_nuclei(self):
        nuc_id_mask = self.output_pack[self.nuc_class]
        cyto_id_mask = self.output_pack[self.cyto_class]
        all_cytos = np.unique(cyto_id_mask)
        linked_nucs = np.unique(nuc_id_mask)
        updated_cyto_id_mask = cyto_id_mask.copy()
        for cyto_id in all_cytos:
            if cyto_id == 0:
                continue
            if cyto_id not in linked_nucs:
                updated_cyto_id_mask = np.where(updated_cyto_id_mask == cyto_id, 0, updated_cyto_id_mask)
        self.output_pack[self.cyto_class] = updated_cyto_id_mask

    def reset_id_sequence(self):
        nuc_id_mask = self.output_pack[self.nuc_class]
        cyto_id_mask = self.output_pack[self.cyto_class]
        all_cytos = np.unique(cyto_id_mask)
        all_nucs = np.unique(nuc_id_mask)
        updated_cyto_id_mask = cyto_id_mask.copy()
        updated_nuc_id_mask = nuc_id_mask.copy()
        new_id = 1
        for cell_id in all_cytos:
            if cell_id == 0:
                continue
            updated_cyto_id_mask = np.where(updated_cyto_id_mask == cell_id, new_id, updated_cyto_id_mask)
            updated_nuc_id_mask = np.where(updated_nuc_id_mask == cell_id, new_id, updated_nuc_id_mask)
            new_id += 1
        self.output_pack[self.cyto_class] = updated_cyto_id_mask
        self.output_pack[self.nuc_class] = updated_nuc_id_mask

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

    def plot_output(self):
        pass

    def save_id_masks(self):
        nuc_id_mask = self.output_pack[self.nuc_class]
        cyto_id_mask = self.output_pack[self.cyto_class]
        nuc_id_mask_path = self.id_mask_path + "nuc_id_mask.npy"
        cyto_id_mask_path = self.id_mask_path + "cyto_id_mask.npy"
        np.save(nuc_id_mask_path, nuc_id_mask)
        np.save(cyto_id_mask_path, cyto_id_mask)

    def save_output(self):
        self.save_id_masks()
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
        nuc_segment_img = ip.resize_img(
            nuc_segment_img,
            base_height,
            base_width,
            "linear")
        cyto_segment_img = ip.resize_img(
            cyto_segment_img,
            base_height,
            base_width,
            "linear")
        segment_img = nuc_segment_img + cyto_segment_img
        segment_overlay = base_img.copy()*0.5 + segment_img*0.5

        nuc_contour = ip.generate_advanced_contour_img(nuc_id_mask)
        cyto_contour = ip.generate_advanced_contour_img(cyto_id_mask)
        nuc_contour = ip.resize_img(
            nuc_contour,
            base_height,
            base_width,
            "linear")
        cyto_contour = ip.resize_img(
            cyto_contour,
            base_height,
            base_width,
            "linear")
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

        seg_path = self.cell_figures_folder + "manual_cell_segment.png"
        self.save_img(segment_img, seg_path)
        seg_over_path = self.cell_figures_folder + "manual_cell_overlay.png"
        self.save_img(segment_overlay, seg_over_path)
        outline_path = self.cell_figures_folder + "manual_cell_outline.png"
        self.save_img(outline_img, outline_path)
        seg_over_act_path = self.cell_figures_folder + "manual_cell_overlay_activated.png"
        self.save_img(segment_overlay_activated, seg_over_act_path)
        outline_act_path = self.cell_figures_folder + "manual_cell_outline_activated.png"
        self.save_img(outline_activated, outline_act_path)
        nuc_seg_path = self.cell_figures_folder + "manual_nuc_segment.png"
        self.save_img(nuc_segment_img, nuc_seg_path)
        cyto_seg_path = self.cell_figures_folder + "manual_cyto_segment.png"
        self.save_img(cyto_segment_img, cyto_seg_path)

class MSSGui():
    """
    The GUI object. Handles everything to do with displaying an image, 
    rectangles, removal of rectangles, and reporting rectangle coordinates to
    the processing node. Has 3 stages. Nucleus Segmentation, Cytoplasm 
    Segmentation, background save image selection.
    

    Global Variables:
    Global Functions:
    Attributes:
        channel_states (dict): flag for whether a channel has been selected or
        not
        z_states (dict): flag for whether a z axis has been selected or not
        canv_width (int): width of canvas
        canv_height (int): height of canvas
        app_width (int): width of app
        app_height (int): height of app
        images_reps (int): the number of stages. 3 total
        curr_rep (int): current stage
        master_node (ManualSamCellSegmenter): the node that created this GUI
        root (Tk): root object for tkinter
        curr_img (PhotoImage): current displayed image
        canvas (Canvas): canvas object
        rect (RectTracker): RectTracker object
        image_container (Tkinter Image): realizes the image in the canvas
        button_frame (Frame): frame that contains all buttons
        continue_button (Button): when clicked continues to next stage
        reset_button (Button): when clicked resets all rectangles and image
        segment_view_button (Button): when clicked shows segmentations
        default_view_button (Button): when clicked shows image and outlines
        segment_button (Button): segments the image given current boxes
        nuc_overlay_btn (Button): when clicked overlays segmented nuclei
        channel_buttons (Button): buttons pertaining to each channel
        z_buttons (Button): buttons pertaining to each z axis
        curr_view (str): string that specifies what the current view is
        overlay_with_nuc_seg (boolean): state of the nucleus overlay
        
    Methods:
        z_adjustment(btn_name): when a z button is pressed change its color
        and change the image to now include or exclude the z axis
        chan_adjustment(btn_name): when a channel button is pressed change
        its color and change the image to now include or exclude the 
        channel
        nuc_overlay(): overlays the image with nucleus segmentations. Only
        works during cytoplasm stage
        get_bboxes(): returns all rectangle bboxes
        remove_tiny_bboxes(bboxes): removes bboxes that are smaller than a
        minimum value
        segment(): using the current state segments the image
        refresh_view(): refreshes the view
        reset(): resets the stage
        continue_program(): moves on to the next stage
        get_mask_class_from_user(): returns what object the stage is segmenting
        exit_gui(): destroys the gui
        remove_all_boxes(): removes all rectangles from canvas
        segment_view(): display segmentation view
        default_view(): display the default view
        segment_box(): segment a single box
        run(): runs the GUI
        update_img(img_arr): updates the image within the Canvas
        on_click(event): if cursor is within a rectangle and right click is 
        pressed deletes the rectangle
        on_mouse_over(event): highlights a rectangle as red if its highlighted
        otherwise keeps them black
    """
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
        self.canvas = tk.Canvas(self.root,
            width=self.canv_width,
            height=self.canv_height)
        self.canvas.pack()
        self.rect = RectTracker(self.canvas, self, self.box_tag)
        def on_drag(start, end):
            self.rect.get_box(start, end)
        self.rect.autodraw(fill="", width=2, command=on_drag)
        
        self.img_container = self.canvas.create_image(
            0,
            0,
            anchor="nw",
            image=self.curr_img)

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

    # probably a way to combine z_adjustment and chan_adjustment into one
    # method
    def z_adjustment(self, btn_name):
        """
        When a z button is clicked change the color to reflect whether its
        being turned on or off(green for on, red for off). Then report the
        updated z states and channel states to the ManualSamCellSegmenter
        to get an updated image. Finally reset the stage under the new
        image context
        
        Args:
            btn_name (str): name of the button that was pressed

        Returns:
            Nothing
        """
        btn = self.z_buttons[btn_name]
        self.z_states[btn_name] = not self.z_states[btn_name] 
        if self.z_states[btn_name]:
            btn.configure(bg = "green")
        else:
            btn.configure(bg = "red")
        self.master_node.reinitialize_base_img(
            self.channel_states,
            self.z_states)
        self.reset()

    def chan_adjustment(self, btn_name):
        """
        When a channel button is clicked change the color to reflect whether its
        being turned on or off(green for on, red for off). Then report the
        updated z states and channel states to the ManualSamCellSegmenter
        to get an updated image. Finally reset the stage under the new
        image context
        
        Args:
            btn_name (str): name of the button that was pressed

        Returns:
            Nothing
        """
        btn = self.channel_buttons[btn_name]
        self.channel_states[btn_name] = not self.channel_states[btn_name] 
        if self.channel_states[btn_name]:
            btn.configure(bg = "green")
        else:
            btn.configure(bg = "red")
        self.master_node.reinitialize_base_img(
            self.channel_states,
            self.z_states)
        self.reset()

    def nuc_overlay(self):
        """
        Displays the nucleus segmentations processed in the nucleus 
        segmentation stage as an overlay. Changes the color to reflect
        if this buttons is on or off
        Only works during the cytoplasm step.
        
        Args:
            Nothing

        Returns:
            Nothing
        """
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
        """
        Collects the bounding boxes from all currently drawn rectangles
        and returns them
        
        Args:
            Nothing

        Returns:
            list: bboxes
        """
        bboxes = []
        boxes = []
        boxes.extend(self.canvas.find_withtag(self.box_tag))
        for box in boxes:
            bboxes.append(self.canvas.coords(box))
        return bboxes

    def remove_tiny_bboxes(self, bboxes):
        """
        Given a list of bboxes removes the ones that are smaller than the
        minimum area. This is to help deal with situations where a user 
        draws a rectangle too small to delete.
        
        Args:
            bboxes (list): list of bboxes

        Returns:
            list: bboxes that are larger than min area
        """
        min_area = 100
        bboxes_pruned = []
        for bbox in bboxes:
            bbox_area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
            if bbox_area > min_area:
                bboxes_pruned.append(bbox)
        return bboxes_pruned
          

    def segment(self):
        """
        Requests the ManualSamCellSegmenter to take the current bboxes 
        and image and produces a segmentation. After segmentation is
        produced refresh the view.
        Only works during stage 0 and 1 (nucleuss and cytoplasm segmentation
        step)
        
        Args:
            Nothing

        Returns:
            Nothing
        """
        if self.curr_rep == 2:
            print("This button only works on a segmentation step")
            return
        bboxes = self.get_bboxes()
        bboxes = self.remove_tiny_bboxes(bboxes)
        self.master_node.update_bboxes(bboxes)
        self.master_node.process_img()
        self.refresh_view()

    def refresh_view(self):
        """
        Refreshes the view with the updates segmentation data.
        
        Args:
            Nothing

        Returns:
            Nothing
        """
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
        """
        Removes all rectangles and reverts the image to its base state before
        any segmentations.
        
        Args:
            Nothing

        Returns:
            Nothing
        """
        self.master_node.soft_reset()
        self.refresh_view()
        self.remove_all_boxes()

    def continue_program(self):
        """
        Moves on to the next stage. This involves reseting the image,
        rectangles, and title. After stage 2(the final image selection
        step) the gui is considered finished and closes itself.
        
        Args:
            Nothing

        Returns:
            Nothing
        """
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
        """
        Returns the cell part being processed by the current stage
        
        Args:
            Nothing

        Returns:
            str: cell part being processed
        """
        if self.curr_rep == 0:
            return "nucleus"
        elif self.curr_rep == 1:
            return "cytoplasm"

    def exit_gui(self):
        """
        Closes the GUI
        
        Args:
            Nothing

        Returns:
            Nothing
        """
        self.root.destroy()

    def remove_all_boxes(self):
        """
        Removes all user drawn rectangles/boxes 
        
        Args:
            Nothing

        Returns:
            Nothing
        """
        boxes = []
        boxes.extend(self.canvas.find_withtag(self.box_tag))
        for box in boxes:
            self.canvas.delete(box)
        
    def segment_view(self):
        """
        Displays the colored segmentation view. Does not work on the background
        image selection stage (stage 2).
        
        Args:
            Nothing

        Returns:
            Nothing
        """
        if self.curr_rep == 2:
            print("This button only works on a segmentation step")
            return
        self.curr_view = "segment"
        self.refresh_view()
        
    def default_view(self):
        """
        Displays the default view which is the image with outlines if a 
        segmentation was performed.
        
        Args:
            Nothing

        Returns:
            Nothing
        """
        self.curr_view = "default"
        self.refresh_view()

    def segment_box(self, box):
        """
        NOT DEFINED
        Segments a single box
        
        
        Args:
            box (list): bounding box

        Returns:
            Nothing
        """
        pass

    def run(self):
        """
        Starts the GUI
        
        Args:
            Nothing

        Returns:
            Nothing
        """
        self.root.mainloop()

    def update_img(self, img_arr):
        """
        Updates the displayed image using the argument image
        
        Args:
            img_arr (ndarray): numpy array containg image data

        Returns:
            Nothing
        """
        img_arr = img_arr.astype(np.uint8)
        self.curr_img =  ImageTk.PhotoImage(image=Image.fromarray(img_arr))
        self.canvas.itemconfig(self.img_container, image=self.curr_img)

    def on_click(self, event):
        """
        When a right click event occurs check to see if the cursor is within
        a rect. If it is then delete the rectangle.
        
        Args:
            event (Tkinter Event): right mouse button click

        Returns:
            Nothing
        """
        x = event.x
        y = event.y
        selected_rect = self.rect.mouse_hit_test([x,y], tags=[self.box_tag])
        if selected_rect is not None:
            self.canvas.delete(selected_rect)

    def on_mouse_over(self, event):
        """
        When the cursor mouses over a rectangle then change the color of the
        rectangle to be red. Return to black when the mouse leaves.

        Args:
            event (Tkinter Event): right mouse button click

        Returns:
            Nothing
        """
        x = event.x
        y = event.y
        selected_rect = self.rect.mouse_hit_test([x,y], tags=[self.box_tag])
        for sub_rect in self.rect.items:
            if sub_rect is not selected_rect:
                self.canvas.itemconfig(sub_rect, outline='black')
            else:
                self.canvas.itemconfig(sub_rect, outline='red')

class RectTracker:
    """
    Responsible for what happens before, during, and after a rectangle is drawn.
    General behavior is a rectangle is drawn when the left mouse button is 
    held in the frame. The rectangles generated by this process are only
    temprorary for visual feedback. Once the left mouse button is let go
    then final rectangle becomes "permanent" as it is not deleted. These
    "permanent" rectangles are interactable and can be destroyed by 
    right clicking them. When the mouse is hovered inside a "permanent"
    rectangle it turns the rectangle red as feedback to the user.

    Global Variables:
    Global Functions:
    Attributes:
        canvas (Tkinter Canvas): the parent canvas object
        gui (Tkinter GUI): the parent gui object
        box_tag (str): tag class of box
        item (int): tkinter identifier for item
        box (list): box coordinates
    Methods:
        __update(event): draws an expanding selection rectangle
        __stop(event): stops drawing an expanding selection rectangle
        draw(start, end): draws a rectangle given the coordinates
        autodraw(): overall event handling logic for drawing selection rects
        get_box(start, end): sets the given coordinates as the rectangles
        current coordinates
        mouse_hit_test(pos): checks to see if a cursor is within a rectangle
    """
    def __init__(self, canvas, gui, box_tag):
        self.canvas = canvas
        self.gui = gui
        self.box_tag = box_tag
        self.item = None
        self.box = None

    def __update(self, event):
        """
        Updates a rectangles during the selection process of drawing a 
        selection rectangle. The event being considered is a mouse
        moving while the left button is clicked. This is done by initially
        creating the rectnagle if start is None. Then by deleting the 
        previously drawn rectangle and then drawn the new one.

        Args:
            event (Tkinter Event): mouse moving while left clicked event

        Returns:
            Nothing
        """
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
        """
        When the left mouse button is released set the start and rectangle
        attributes to None. This ends the selection window process.

        Args:
            event (Tkinter Event): release of the left mouse button

        Returns:
        """
        self.start = None
        self.item = None
		
    def draw(self, start, end, **opts):
        """
        Given a start and end coordinates draws a rectangle in the canvas.

        Args:
            start (list): x and y coordinates of when the box was initially drawn
            end (list): x and y coordinates of the cursor

        Returns:
            int: id of rectangle
        """
        return self.canvas.create_rectangle(*(list(start)+list(end)), **opts)
		
    def autodraw(self, **opts):
        """
        Binds mouse events to methods within to realize the box selection
        functionality.

        Args:
            Nothing

        Returns:
            Nothing
        """
        self.start = None
        self.canvas.bind("<Button-1>", self.__update, '+')
        self.canvas.bind("<B1-Motion>", self.__update, '+')
        self.canvas.bind("<ButtonRelease-1>", self.__stop, '+')
        self._command = opts.pop('command', lambda *args: None)
        self.rectopts = opts
	
    def get_box(self, start, end, tags=None, ignoretags=None, ignore=[]):
        """
        Sets the box attribute with the given coordinates
        Current method name should be changed in the future to match
        functionality.
        

        Args:
            start (list): x and y coordinates of when the box was initially drawn
            end (list): x and y coordinates of the cursor

        Returns:
            Nothing
        """
        xlow = min(start[0], end[0])
        xhigh = max(start[0], end[0])
	
        ylow = min(start[1], end[1])
        yhigh = max(start[1], end[1])
	
        self.box = [xlow, ylow, xhigh, yhigh]

    def mouse_hit_test(self, pos, tags=None, ignoretags=None, ignore=[]):
        """
        Deals with the logic of checking to see if the cursor is within a 
        tkinter object. Handles the cast of when objects are within object
        by selecting the object with smallest area first. 

        Args:
            pos (list): x and y coordinates of cursor
            tags (str): tag of tkinter object that is being considered

        Returns:
            int: id of object mouse is within
        """
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

    def give_final_box(self):
        """

        Args:

        Returns:
        """
        pass
