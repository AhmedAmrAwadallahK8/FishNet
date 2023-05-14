import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import src.user_interaction as usr_int
from src.nodes.AbstractNode import AbstractNode
from src.fishnet import FishNet





class SimpleNucleusCounter(AbstractNode):
    def __init__(self):
        super().__init__(output_name="SimpleNucleusCount",
                         requirements=["NucleusMask"],
                         user_can_retry=False,
                         node_title="Simple Nucleus Counter")
        pass

    def initialize(self):
        pass

    def process(self):
        pass
        
