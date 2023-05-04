from nd2reader import ND2Reader
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from src.common import TempPipeline

class SampleNode():
   def __init__(self):
      self.output_name = "SampleName"

   def get_output_name(self):
      return self.output_name

   def process(self):
      return 0

class FishNet():
   raw_imgs = []
   pipeline_output = {}
   def __init__(self):
      self.placeholder = 0
      self.version = 0.01
      self.valid_file_types = ["nd2"]
      self.img_file = ""
      # self.all_imgs = []
      self.nodes = [SampleNode()]
      self.pipeline = TempPipeline(self.nodes)
      self.valid_responses = ["yes", "y", "no", "n"]
      self.negative_responses = ["no", "n"]
      self.positive_responses = ["yes", "y"]

      self.invalid_response_id = 0
      self.positive_response_id = 1
      self.negative_response_id = 2

      self.welcome_message = f"Welcome to FishNet v{self.version}!"
      self.goodbye_message = f"Thank you for using FishNet! Goodbye."

   def run(self):
      self.welcome()
      self.prompt_user_for_file()
      self.extract_img_info()
      self.run_pipeline()
      self.goodbye()

   def welcome(self):
      print(self.welcome_message)
   
   # Assuming that a node outputs exactly what the next node wants
   def run_pipeline(self):
      pipeline_advanced = True
      while(self.pipeline.is_not_finished()):
         output_img, img_name = self.pipeline.process_node()
         user_satisfied = self.check_if_user_satisified(output_img)
         if user_satisfied:
            self.store_output_img(output_img, img_name)
            pipeline_img = output_img
            self.pipeline.advance()
         else:
            user_wants_to_retry = self.ask_user_to_try_again_or_quit()
            if user_wants_to_retry:
               continue
            else:
               break


   def process_user_input(self, user_input):
      user_input = user_input.lower()
      if user_input in self.valid_responses:
         if user_input in self.positive_responses:
            return self.positive_response_id
         elif user_input in self.negative_responses:
            return self.negative_response_id
      else:
         return self.invalid_response_id


   def ask_user_to_try_again_or_quit(self):
      prompt = "Would you like to try this step again?\n"
      prompt += "If you say no the program will assume you are done and exit. "
      user_input = input(prompt)
      response_id = self.process_user_input(user_input)
      if response_id == self.positive_response_id:
         return True
      elif response_id == self.negative_response_id:
         return False

   def check_if_user_satisified(self, output_img):
      # Display Img
      prompt = "Are you satisfied with the displayed image"
      prompt += " for this step? "
      response_id = self.invalid_response_id
      while(response_id == self.invalid_response_id):
         user_input = input(prompt)
         response_id = self.process_user_input(user_input)
         if response_id == self.positive_response_id:
            return True
         elif response_id == self.negative_response_id:
            return False
         elif response_id == self.invalid_response_id:
            print("Invalid response try again.")
            print("We expect either yes or no.")

   def goodbye(self):
      print(self.goodbye_message)

   def store_output_img(self, img, img_name):
      FishNet.pipeline_output[img_name] = img

   def prompt_user_for_file(self):
      self.img_file = input("Input file to be processed: ")

   def extract_img_info(self):
      with ND2Reader(self.img_file) as images: 
         images.iter_axes = 'zc'
         z_stack = int(input("Specify how many z slices: "))
         c_stack = int(input("Specify how many experiment channels: "))
         FishNet.raw_imgs = []
         for z in range(z_stack):
            FishNet.raw_imgs.append([])
            for c in range(c_stack):
               FishNet.raw_imgs[z].append(images[z*c_stack + c])

if __name__ == '__main__':
   f = FishNet()
   f.run()
