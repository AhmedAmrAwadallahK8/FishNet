import numpy as np
import cv2
import cv2 as cv
from src.nodes.AbstractNode import AbstractNode
import src.image_processing as ip
import src.user_interaction as usr_int
import os

class CellMeanIntensity(AbstractNode):
    """
    Node that sums the area and intensity within cells selected by
    ManualSamCellSegmenter. These values are seperated by nucleus
    and cytoplasm.
    Object name may be slightly misleading as we don't directly compute the
    mean for the user we simple provide the values necessary for them to 
    do so in any way they prefer

    Global Variables:
    Global Functions:
    Attributes:
        base_img (ndarray): current image being processed
        cyto_id_mask (ndarray): cytoplasm id mask
        nuc_id_mask (ndarray): nucleus id mask
        cytoplasm_key (str): key to access cytoplasm data in various dicts
        nucleus_key (str): key to access nucleus data in various dicts
        cell_id_mask (ndarray): cell id mask
        csv_name (str): name of csv file
        nuc_area (dict): stores area of nucleus by cell id
        cyto_area (dict): stores area of cytoplasm by cell id
        nuc_intensity_sum (dict): stores intensity sum of nucleus by cell id
        cyto_intensity_sum (dict): stores intensity sum of cytoplasm by cell id
        csv_data (list): list of csv structured observations
        z_key (str): key to access z related metadata
        c_key (str): key to access channel related metadata
        z (str): current z axis
        c (str): current channel
        settings (list): all keys associated with settings that are 
        interactable by the user
        user_settings (dict): settings that user selected
        setting_type (dict): general data type of a setting
        setting_range (dict): set of acceptable values per setting
        setting_description (dict): description of setting given to user 
        prior to asking for input
    Methods:
        process(): performs all critical actions of this node
        process_cell_part(cell_part): processes a cell part
        initialize_node(): does all important runtime setup prior to running
        this node
        get_setting_selections_from_user(): input loop to get desired settings
        from user
        finish_setting_setup(): setup step that can only occur at run time of
        node
        get_id_mask(): gets id mask from FishNet
        save_output(): writes important output to disk
        update_context_img(): updates the context image that is being 
        processed
        store_csv_data(): locally stores csv structured data
        save_csv(): writes csv data to disk
        get_area_and_intensity_sum(img_crop): returns the area and intensity
        sum of the given crop
        get_segmentation_bbox(single_id_mask): returns bounding box around
        a single id mask
        calc_mean_intensity(img_crop): not currently used but remains for now
    """
    def __init__(self):
        from src.fishnet import FishNet
        super().__init__(output_name="CellMeanIntensityPack",
                         requirements=["ManualCellMaskPack"],
                         user_can_retry=False,
                         node_title="Cell Mean Intensity")
        self.base_img = None
        self.cyto_id_mask = None
        self.nuc_id_mask = None
        self.cytoplasm_key = "cyto"
        self.nucleus_key = "nuc"
        self.cell_id_mask = None
        self.csv_name =  "cell_intensity.csv"
        self.nuc_area = {}
        self.cyto_area = {}
        self.nuc_intensity_sum = {}
        self.cyto_intensity_sum = {}
        self.csv_data = []

        self.z_key = "Z Axis"
        self.c_key = "Channel Axis"
        self.z = None
        self.c = None

        self.settings = [
            self.z_key,
            self.c_key]
        self.user_settings = {}
        self.setting_type = {
            self.z_key: "categ",
            self.c_key: "categ"}
        self.setting_range = {}
        self.setting_description = {}

    def process(self):
        """
        Performs all critical actions of this node. This pertains to going 
        through each cell at each z and channel index combination and retrieving
        the pixel area and intensity sum.

        Args:
            Nothing

        Returns:
            Nothing
        """
        print(f"Percent Done: 0.00%")
        process_count = 0 
        total_count = len(self.user_settings[self.z_key])
        total_count *= len(self.user_settings[self.c_key])
        for z_axis in self.user_settings[self.z_key]:
            for c_axis in self.user_settings[self.c_key]:
                process_count += 1
                self.z = z_axis
                self.c = c_axis
                self.update_context_img()
                self.process_cell_part(self.cytoplasm_key)
                self.process_cell_part(self.nucleus_key)
                self.store_csv_data()
                percent_done = process_count/total_count*100
                print(f"Percent Done: {percent_done:.2f}%")
        self.set_node_as_successful()

    def process_cell_part(self, cell_part):
        """
        Processes a specific cell part(nucleus or cytoplasm). This involves
        isolating the cell part then getting and storing its intensity sum
        and area.

        Args:
            cell_part (str): specifies part of cell being processed

        Returns:
            Nothing
        """
        id_mask = self.cell_id_mask[cell_part]
        cell_ids = np.unique(id_mask)
        for cell_id in cell_ids:
            if cell_id == 0 or cell_id > self.max_cell_id:
                continue

            targ_shape = self.base_img.shape
            id_activation = np.where(id_mask == cell_id, 1, 0)
            resized_id_activation = ip.resize_img(
                id_activation,
                targ_shape[0],
                targ_shape[1],
                inter_type="linear")
            id_bbox = self.get_segmentation_bbox(id_activation)
            id_bbox = ip.rescale_boxes(
                [id_bbox],
                id_activation.shape,
                self.base_img.shape)[0]
            xmin = int(id_bbox[0])
            xmax = int(id_bbox[2])
            ymin = int(id_bbox[1])
            ymax = int(id_bbox[3])
            img_id_activated = resized_id_activation * self.base_img
            img_crop = img_id_activated[ymin:ymax, xmin:xmax].copy()
            area, intensity_sum = self.get_area_and_intensity_sum(img_crop)
            if cell_part == self.cytoplasm_key:
                self.cyto_area[cell_id] = area
                self.cyto_intensity_sum[cell_id] = intensity_sum
            elif cell_part == self.nucleus_key:
                self.nuc_area[cell_id] = area
                self.nuc_intensity_sum[cell_id] = intensity_sum

    def initialize_node(self):
        """
        Performs all critical functions related to initializing this node.
        This involves get mask ids and getting setting selections from user

        Args:
            Nothing

        Returns:
            Nothing
        """
        self.finish_setting_setup()
        self.get_id_mask()
        self.get_setting_selections_from_user()

    def get_setting_selections_from_user(self):
        """
        Prompts user for a response for a particular setting and stores their
        response within user_settings

        Args:
            Nothing

        Returns:
            Nothing
        """
        print("")
        for setting in self.settings:
            user_setting = None
            setting_message = self.setting_description[setting]
            print(setting_message)
            msg = f"Input a value for {setting}: "
            setting_range = self.setting_range[setting]
            if self.setting_type[setting] == "categ":
                user_setting = usr_int.get_categorical_input_set_in_range(
                    msg,
                    setting_range)
            self.user_settings[setting] = user_setting
            print("")

    def finish_setting_setup(self):
        """
        Finalize the setup of the user settings. Happens at node process time
        since some information for settings is not available

        Args:
            Nothing

        Returns:
            Nothing
        """
        from src.fishnet import FishNet
        self.setting_range = {
            self.z_key: list(FishNet.z_meta.keys()),
            self.c_key: list(FishNet.channel_meta.keys())}
        z_descrip = f"Enter all the z axi that you are interested "
        z_descrip += f"in seperated by commas\n"
        z_descrip += f"Valid z axi are in {self.setting_range[self.z_key]}."

        c_descrip = f"Enter all the channels that you are interested "
        c_descrip += f"in seperated by commas\n"
        c_descrip += f"Valid channels are in {self.setting_range[self.c_key]}."
        self.setting_description = {
            self.z_key: z_descrip,
            self.c_key: c_descrip}

    def get_id_mask(self):
        """
        Gets the relevant id masks outputted by the Manual Sam Cell node
        from the FishNet pipeline_output variable

        Args:
            Nothing

        Returns:
            Nothing
        """
        from src.fishnet import FishNet
        mask_pack = FishNet.pipeline_output["ManualCellMaskPack"]
        self.cyto_id_mask = mask_pack["cytoplasm"]
        self.nuc_id_mask = mask_pack["nucleus"]
        self.cell_id_mask = {
            self.cytoplasm_key: self.cyto_id_mask,
            self.nucleus_key: self.nuc_id_mask
        }
        self.max_cell_id = np.max(self.cyto_id_mask)

    def save_output(self):
        """
        Writes the csv structured data to disk

        Args:
            Nothing

        Returns:
            Nothing
        """
        self.save_csv()

    def update_context_img(self):
        """
        Updates the current context image based on the current channel and
        z index

        Args:

        Returns:
        """
        from src.fishnet import FishNet
        c_ind = FishNet.channel_meta[self.c]
        z_ind = FishNet.z_meta[self.z]
        self.base_img = ip.get_specified_channel_combo_img([c_ind], [z_ind])


    def store_csv_data(self):
        """
        Store csv structured data locally in csv_data

        Args:
            Nothing

        Returns:
            Nothing
        """
        for cell_id in self.nuc_intensity_sum.keys():
            obs = f"{cell_id},{self.cyto_intensity_sum[cell_id]:.1f},"
            obs += f"{self.cyto_area[cell_id]:.1f},"
            obs += f"{self.nuc_intensity_sum[cell_id]:.1f},"
            obs += f"{self.nuc_area[cell_id]:.1f},{self.z},{self.c}\n"
            self.csv_data.append(obs)

    def save_csv(self):
        """
        Writes the csv structured data in csv_data to disk

        Args:
            Nothing

        Returns:
            Nothing
        """
        from src.fishnet import FishNet
        # csv of particle counts
        csv_path = FishNet.save_folder + self.csv_name
        csv_file = open(csv_path, "w")
        csv_header = "cell_id,cyto_intensity_sum,cyto_area,nuc_intensity_sum,"
        csv_header += "nuc_area,z_level,channel\n"
        csv_file.write(csv_header)
        for obs in self.csv_data:
                csv_file.write(obs)
        csv_file.write("\n")
        csv_file.close()


    def get_area_and_intensity_sum(self, img_crop):
        """
        Using an activated image crop returns the area and intensity sum of
        the activation. Only processes area that has a value greater than 0

        Args:
            img_crop (ndarray): activated image crop of a cell part

        Returns:
            float: pixel area
            float: intensity sum
        """
        area = np.sum(np.where(img_crop > 0, 1, 0))
        intensity_sum = np.sum(img_crop)
        return area, intensity_sum

    def calc_mean_intensity(self, img_crop):
        """
        Calculates the mean intensity of an activated image crop. Only considers
        pixels that are non zero.

        Args:
            img_crop (ndarray): activated image crop of a cell part

        Returns:
            float: mean intensity
        """
        total_pix = np.sum(np.where(img_crop > 0, 1, 0))
        intensity_sum = np.sum(img_crop)
        mean_intensity = intensity_sum/total_pix
        return mean_intensity

    def get_segmentation_bbox(self, single_id_mask):
        """
        Takes in a mask that contains only one unique id within it and outputs
        a bounding box around the segmentation. Single id masks can have 
        seperated segmentations so parse for the segmentation with the largest
        area first

        Args:
            single_id_mask (ndarray): mask with only one unique id present
        Returns:
            list: bounding box around mask
        """
        gray = single_id_mask[:, :, np.newaxis].astype(np.uint8)
        contours, hierarchy = cv2.findContours(
            gray,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE)[-2:]
        idx =0 
        rois = []
        largest_area = 0
        best_bbox = []
        first = True
        for cnt in contours:
            idx += 1
            area = cv.contourArea(cnt)
            rect_pack = cv2.boundingRect(cnt) #x, y, w, h
            x, y, w, h = rect_pack
            bbox = [x, y, x+w, y+h]
            if first:
                first = False
                largest_area = area
                best_bbox = bbox
            else:
                if area > largest_area:
                    largest_area = area
                    best_bbox = bbox
        return best_bbox
