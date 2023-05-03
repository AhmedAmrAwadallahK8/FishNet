from nd2reader import ND2Reader
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

class FishNet():
   def __init__(self):
      self.placeholder = 0
      self.version = 0.01
      self.valid_file_types = ["nd2"]
      self.img_file = ""

      self.welcome_message = f"Welcome to FishNet v{self.version}!"
      self.goodbye_message = f"Thank you for using FishNet! Goodbye."

   def run(self):
      self.welcome()
      self.prompt_user_for_file()
      self.extract_img_info()

      
      self.goodbye()

   def welcome(self):
      print(self.welcome_message)

   def check_file_exists(self):
      return 0

   def goodbye(self):
      print(self.goodbye_message)

   def prompt_user_for_file(self):
      self.img_file = input("Input file to be processed: ")

   def extract_img_info(self):
      all_imgs = [[],[],[],[],[]]
      with ND2Reader(self.img_file) as images: 
         images.iter_axes = 'zc'
         z_stack = int(input("Specify how many z slices: "))
         c_stack = int(input("Specify how many experiment channels: "))
         for z in range(z_stack):
            for c in range(c_stack):
               all_imgs[z].append(images[z*c_stack + c])





if __name__ == '__main__':
   f = FishNet()
   f.run()
