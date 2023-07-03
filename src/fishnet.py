from nd2reader import ND2Reader
import cv2 as cv
import numpy as np
import src.user_interaction as usr_int
import sys
import os
import shutil
from src.common import TempPipeline
from src.nodes.ManualSamCellSegmenter import ManualSamCellSegmenter
from src.nodes.SamCellDotCounter import SamCellDotCounter
from src.nodes.CellMeanIntensity import CellMeanIntensity
from src.wrappers.local_sam import LocalSam

class FishNet():
   """
   Responsible for the overall flow of the program and a global data storage
   for other objects within the program to communcicate

   Global Attributes:
      sam_model (LocalSam): Wrapper object for SAM
      raw_imgs (list): Contains all the nd2 image slices
      channel_meta (dict): Channel metadata from nd2 file
      z_meta (dict): Z axis meta data from nd2 file
      pipeline_output (dict): Global storage for pipeline node output
      save_folder (str): Output folder name

   Attributes:
      version (int): Version number of program 
      valid_file_types (list): List of valid files
      img_file (str): Path of nd2 file
      pipeline (TempPipeline): Main pipeline object
      welcome_message (str): Program welcome message
      goodbye_message (str): Program goodbye message

   Global Function:
      store_output(output, out_name): allows objects to request to store data
      within the pipeline_output global attribute

   Methods:
      run(): Performs all critical actions of class
      welcome(): Prints welcome message
      user_exit(): Prints exit message then terminates program
      run_pipeline(): Request pipeline to sequential process
      goodbye(): Prints goodbye message
      prompt_user_for_file(): Request user to input nd2 file path and stores
      in img_file attribute
      extract_img_info(): Extracts nd2 file for all imgs and metadata associated
      with the z axis and color channels. Stores relevant information in the 
      raw_imgs, channel_meta, and z_meta attributes
      
   """

   sam_model = LocalSam()
   raw_imgs = []
   channel_meta = {}
   z_meta = {}
   pipeline_output = {}
   save_folder = "output/"

   def store_output(output, out_name):
      """
      Takes in a key and value and stores it within the global pipeline_output 
      dictionary. Only used by nodes to store output if they information
      other nodes would find useful.

      Args:
         output (any): Useful node output
         out_name (str): Key to reference the node output
      
      Returns:
         Nothing
      """
      FishNet.pipeline_output[out_name] = output

   def __init__(self):
      if not os.path.exists(FishNet.save_folder):
          os.makedirs(FishNet.save_folder)
      elif os.path.exists(FishNet.save_folder):
          shutil.rmtree(FishNet.save_folder)
          os.makedirs(FishNet.save_folder)
      self.version = 0.01
      self.valid_file_types = ["nd2"]
      self.img_file = ""
      self.pipeline = TempPipeline([
         ManualSamCellSegmenter(),
         CellMeanIntensity(),
         SamCellDotCounter()])

      self.welcome_message = f"Welcome to FishNet v{self.version}!"
      self.goodbye_message = f"Thank you for using FishNet! Goodbye."


   def run(self):
      """
      Runs all critical steps of the program
      
      Args:
         Nothing

      Returns:
         Nothing
      """
      self.welcome()
      self.prompt_user_for_file()
      self.extract_img_info()
      self.run_pipeline()
      self.goodbye()

   def welcome(self):
      """
      Prints the welcome message
      
      Args:
         Nothing

      Returns:
         Nothing
      """
      print(self.welcome_message)

   def user_exit(self):
      """
      Unique program exit path that occurs when a node reports it failed to
      complete its expected work
      
      Args:
         Nothing

      Returns:
         Nothing
      """
      print("An exit input was recieved, program will now terminate.")
      self.goodbye()
      sys.exit()
   
   def run_pipeline(self):
      """
      Iteratively passes through the pipeline and checks nodes for satisfactory
      completition. If node failed to complete it will call user_exit
      
      Args:
         Nothing

      Returns:
         Nothing
      """
      from src.nodes.AbstractNode import AbstractNode
      pipeline_advanced = True
      while(self.pipeline.is_not_finished()):
         node_status_code = self.pipeline.run_node()
         if node_status_code == AbstractNode.NODE_FAILURE_CODE:
             self.user_exit() 
         self.pipeline.advance()

   def goodbye(self):
      """
      Prints goodbye message
      
      Args:
         Nothing

      Returns:
         Nothing
      """
      print(self.goodbye_message)


   def prompt_user_for_file(self):
      """
      Asks user for the nd2 file path and stores within img_file
      
      Args:
         Nothing

      Returns:
         Nothing
      """
      self.img_file = input("Input nd2 path: ")

   def convert_list_to_dict(self, arg_list):
      """
      Converts a list of values into a dictionary where the list values become
      the values of the dictionary and the list index becomes the key
      
      Args:
         arg_list (list): list of values

      Returns:
         dict: dictionary representation of the list
      """
      final_dict = {}
      for i in range(len(arg_list)):
         k = arg_list[i]
         final_dict[k] = i
      return final_dict

   def extract_img_info(self):
      """
      Extracts information from the nd2 file. Specifically all z axis 
      and color channel images as well as relevant metadata.
      
      Args:
         Nothing

      Returns:
         Nothing
      """
      with ND2Reader(self.img_file) as images: 
         channel_info = images.metadata["channels"]
         z_len = len(images.metadata["z_levels"])
         z_info = [str(x) for x in range(1, z_len+1)]
         c_len = len(channel_info)
         
         FishNet.z_meta = self.convert_list_to_dict(z_info)
         FishNet.channel_meta = self.convert_list_to_dict(channel_info)
         images.iter_axes = 'zc'
         FishNet.raw_imgs = []
         for z in range(z_len):
            FishNet.raw_imgs.append([])
            for c in range(c_len):
               FishNet.raw_imgs[z].append(images[z*c_len + c])
      FishNet.raw_imgs = np.asarray(FishNet.raw_imgs)

if __name__ == '__main__':
   f = FishNet()
   f.run()
